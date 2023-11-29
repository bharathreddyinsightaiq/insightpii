from dotenv import load_dotenv
import os
import streamlit as st
import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine, text
import pandas as pd
import urllib.parse
import re
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from azure.storage.blob import BlobServiceClient, BlobClient, generate_blob_sas, BlobSasPermissions
import json
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from io import BytesIO
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Payload, VectorParams, Distance, HnswConfigDiff
from qdrant_client.http import models
from openai import OpenAI
from datetime import datetime, timedelta
import base64

load_dotenv()

# PG Database connection parameters
pg_host = os.getenv('PG_HOST')
pg_port = os.getenv('PG_PORT')
pg_database = os.getenv('PG_DB')
pg_user = os.getenv('PG_USER_NAME')
pg_password = os.getenv('PG_PASSWORD')
encoded_password = urllib.parse.quote(pg_password)
engine = create_engine(f'postgresql+psycopg2://{pg_user}:{encoded_password}@{pg_host}:{pg_port}/{pg_database}')

# SF Database connection parameters
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_RAW_SCHEMA = os.getenv("SNOWFLAKE_RAW_SCHEMA")
SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE")
SNOWFLAKE_LINKED_SCHEMA=os.getenv("SNOWFLAKE_LINKED_SCHEMA")
sf_conn = snowflake.connector.connect(
    user=SNOWFLAKE_USER,
    password=SNOWFLAKE_PASSWORD,
    account=SNOWFLAKE_ACCOUNT,
    warehouse=SNOWFLAKE_WAREHOUSE,
    database=SNOWFLAKE_DATABASE,
    schema=SNOWFLAKE_RAW_SCHEMA
)
sf_conn.cursor().execute(f"USE ROLE {SNOWFLAKE_ROLE}")
sf_conn.cursor().execute(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}")
sf_conn.cursor().execute(f"USE DATABASE {SNOWFLAKE_DATABASE}")
sf_conn.cursor().execute(f"USE SCHEMA {SNOWFLAKE_RAW_SCHEMA}")

#Azure SA
connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
azure_storage_access_key = os.getenv('AZURE_ACCESS_KEY')
#CososDB
COSMOS_ENDPOINT=os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY=os.getenv("COSMOS_KEY")
cosmos_client = CosmosClient(url=COSMOS_ENDPOINT, credential=COSMOS_KEY)
database = cosmos_client.create_database_if_not_exists(id='insightpii')

#azure vision 
vision_key = os.getenv("VISION_KEY")
vision_endpoint = os.getenv("VISION_ENDPOINT")
credentials = CognitiveServicesCredentials(vision_key)
vision_client = ComputerVisionClient(vision_endpoint, credentials)

#azure cognitive services
cog_key = os.getenv("COG_KEY")
cog_endpoint = os.getenv("COG_ENDPOINT")
text_analytics_client = TextAnalyticsClient(endpoint=cog_endpoint, credential=AzureKeyCredential(cog_key))

#qdrant client
qdrant_url = os.getenv("QDRANT_ENDPOINT")
qdrant_key = os.getenv("QDRANT_KEY")
qdrant_client = QdrantClient(
    url=qdrant_url, 
    api_key=qdrant_key,
)

#OpenAI client
openai_api_key = os.getenv("OPENAI_KEY")
client = OpenAI(api_key=openai_api_key)

def get_sas_token(account_name, container_name, blob_name, account_key=azure_storage_access_key):
    sas_token = generate_blob_sas(
        account_name=account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=1)  
    )
    return f'https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}'

def query_qdrant(query, collection_name, top_k=1):
    # Creates embedding vector from user query
    embedded_query = client.embeddings.create(
        input=[query],
        model="text-embedding-ada-002",
    ).data[0].embedding

    query_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=(
            embedded_query
        ),
        limit=top_k,
    )

    return query_results

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def entity_recognition(client, documents):
    try:
        response = client.recognize_pii_entities(documents=documents)
        return response
    except Exception as err:
        print("Encountered exception. {}".format(err))

def extract_text_from_image(image_content_bytes):
    image_stream = BytesIO(image_content_bytes)
    response = vision_client.read_in_stream(image_stream, raw=True)
    read_operation_location = response.headers["Operation-Location"]
    operation_id = read_operation_location.split("/")[-1]
    # Wait for the read operation to complete
    import time
    while True:
        read_result = vision_client.get_read_result(operation_id)
        if read_result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)
    # Extract text
    text_data = ""
    if read_result.status == 'succeeded':
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                text_data += line.text + "\n"
    return text_data

def read_blob(blob_client):
    downloader = blob_client.download_blob()
    return downloader.readall()

def resembles_date(s):
    date_patterns = [
        r'\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}'
    ]
    return any(re.match(pattern, s) for pattern in date_patterns)

def is_parsable_date(s):
    if not resembles_date(s):
        return False
    try:
        pd.to_datetime(s)
        return True
    except:
        return False

def process_dataframe(dataframe):
    new_cols=[]
    for col in dataframe.columns:
        parsable_dates = dataframe[col].astype(str).apply(is_parsable_date)
        # If more than a certain threshold of values can be parsed as dates, consider the column as a date column
        if parsable_dates.mean() > 0.90:  # Adjust the threshold (0.9) as needed
            dataframe[col] = pd.to_datetime(dataframe[col], format='mixed')
            dataframe[col] = pd.to_datetime(dataframe[col]).dt.strftime('%Y-%m-%d')
        new_cols.append(col.replace(' ', '_').upper())
    dataframe.columns = new_cols
    return dataframe

def get_table_creation_sql(dataframe, table_name):
    """Generate a CREATE TABLE SQL statement based on DataFrame dtypes."""
    mapping = {
        "int64": "INTEGER",
        "float64": "FLOAT",
        "object": "TEXT",
        "bool": "BOOLEAN",
        "datetime64[ns]": "TIMESTAMP",
        # Add other types if needed
    }
    columns = []
    for col, dtype in dataframe.dtypes.items():
        sanitized_col = col.replace(' ', '_').upper()
        snowflake_type = mapping.get(str(dtype), "TEXT")
        columns.append(f'"{sanitized_col}" {snowflake_type}')
    columns_sql = ", ".join(columns)
    return f"CREATE OR REPLACE TABLE {table_name} ({columns_sql});"

def upload_to_snowflake(connection, dataframe, file, schema):
    try:
        create_table_sql = get_table_creation_sql(dataframe=dataframe, table_name=file)
        cur = connection.cursor()       
        cur.execute(f"USE SCHEMA {schema}")
        cur.execute(create_table_sql)
        cur.close()
        success, nchunks, nrows, _ = write_pandas(conn=sf_conn, df=dataframe, table_name=file, database=SNOWFLAKE_DATABASE, schema=schema, compression="snappy", quote_identifiers=False)
        st.toast(f'{file} uploaded to Snowflake {schema}.{file}', icon='✅')
    except Exception as e:
        st.write(f"Failed to upload {file}. Error: {e}")

def populate_sf_data(conn, schema):
    dfs = {}
    cur = conn.cursor()
    cur.execute(f"USE SCHEMA {schema}")
    cur.execute("SHOW TABLES")
    tables = [row[1] for row in cur.fetchall()]
    for table in tables:
        cur.execute(f"SELECT * FROM {table}")
        dfs[table] = cur.fetch_pandas_all()
    cur.close()
    return dfs

def clean_cosmos_item(item):
    return {k: v for k, v in item.items() if not k.startswith("_") and k != "id"}

def create_hash(row):
    combined_string = ''.join(row.astype(str))
    hash_object = hashlib.md5(combined_string.encode())
    return hash_object.hexdigest()

def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="350" height="500" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def download_blob_to_file(blob_service_client, container_name, blob_name, file_path):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    with open(file_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())

def scroll_collection_points(collection_name):
    points_data = {}
    scroll_result = qdrant_client.scroll(
        collection_name=collection_name,
        limit=100,  
        with_payload=True,
        with_vectors=True,
    )
    # above returns paginated values with next_page_offset set to null if all results else consult documentation to see how to set offset. 
    if scroll_result[1] is None:
        for result in scroll_result[0]:
            points_data[result.id] = {'vector': result.vector, 'payload': result.payload}
    return points_data

def find_best_matches(current_collection, other_collections, current_points_data):

    best_matches = {}
    
    for point_id, point_info in current_points_data.items():
        point_vector = point_info['vector']
        point_best_matches = {}

        for other_collection in other_collections:
            print_other_coll.add(other_collection)
            search_result = qdrant_client.search(
                collection_name=other_collection,
                query_vector=point_vector,
                limit=1,  
                with_payload=True,
                with_vectors=True
            )[0]

            if search_result:
                best_match = search_result
                point_best_matches[other_collection] = {
                    'match_score': best_match.score,
                    'payload': best_match.payload
                }
        best_matches[point_id] = point_best_matches
    return best_matches

def update_collection_payloads(collection_name, best_matches, existing_payloads):
    for point_id, matches in best_matches.items():
        updated_payload = {**existing_payloads.get(point_id, {}), **matches}
        
        qdrant_client.set_payload(
            collection_name=collection_name,
            payload=updated_payload,
            points=[point_id]
        )


# Page setup
st.set_page_config(page_title="InsightAIQ", layout="wide")
pages = ['Link Records', 'Identify Records', 'Delete Records', 'Data Simulation']
choice = st.sidebar.radio("Choose a page:", pages)

if choice == 'Link Records':
    st.image("Assets/Images/logo.png", width=200)
    st.title('Insight PII')
    st.subheader('PII identification and management solutions')
    st.markdown('##')
    st.write("Seamlessly unify your organization's data sources, effortlessly identify records for individuals, "
            "and even uncover references within unstructured data, documents, and images. Unleash the full potential of "
            "your data integration and record identification needs today!")
    st.subheader('', divider='rainbow')
    st.markdown('##')
    st.subheader("Select Data Sources to link.")
    st.markdown('##')
    full_data={}
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image("Assets/Images/snowflake.png", width=100)
        sf = st.checkbox('Snowflake',key='sf_check')    
        if sf:
            sf_df = populate_sf_data(sf_conn, SNOWFLAKE_RAW_SCHEMA)
            full_data = full_data | sf_df
            st.markdown('##')
            st.write('Snowflake data loaded.')
            st.markdown('##')
            st.metric(label="Snowflake", value=f"{len(sf_df)} Tables", delta=f"{sum(len(df) for df in sf_df.values())} Rows")
    with col2:
        st.image("Assets/Images/postgres.png", width=100)
        pg = st.checkbox('Postgres',key='pg_check')
        if pg:
            schema_name = 'insightpii_raw'
            pg_query = f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema_name}'"
            pg_table_names = pd.read_sql(pg_query, engine)
            tables_dict = {}
            for table_name in pg_table_names['table_name']:
                query = f'SELECT * FROM "{schema_name}"."{table_name}"'
                tables_dict[table_name] = pd.read_sql(query, engine)
            full_data = full_data | tables_dict
            st.markdown('##')
            st.write('Postgres data loaded.')
            st.markdown('##')
            st.metric(label="Postgres", value=f"{len(tables_dict)} Tables", delta=f"{sum(len(df) for df in tables_dict.values())} Rows")
    with col3:
        st.image("Assets/Images/adls.png", width=145)
        sa = st.checkbox('Azure Storage',key='sa_check')
        if sa:
            blob_service_client = BlobServiceClient.from_connection_string(connect_str)
            container_client = blob_service_client.get_container_client(container="adls")
            extracted_text_dict = {}
            temp_df_azure = pd.DataFrame(columns=['text','source'])
            for blob in container_client.list_blobs():
                blob_client = container_client.get_blob_client(blob)
                blob_content = read_blob(blob_client)
                extracted_text = extract_text_from_image(blob_content)
                documents = [extracted_text]
                result = entity_recognition(text_analytics_client, documents)
                entities_info = ""
                if result:
                    for doc in result:
                        if not doc.is_error:
                            for entity in doc.entities:
                                entities_info += f" {entity.text}"
                updated_text = entities_info
                temp_df_azure = pd.concat([temp_df_azure,pd.DataFrame({'text': [updated_text], '_source': blob.name})], ignore_index=True)
                extracted_text_dict['unstructured'] = temp_df_azure
            full_data = full_data | extracted_text_dict  
            st.markdown('##') 
            st.write('Azure Storage data loaded.')
            st.markdown('##')
            st.metric(label="ADLS", value=f"{len(extracted_text_dict)} Tables", delta=f"{sum(len(df) for df in extracted_text_dict.values())} Rows")
    with col4:
        st.image("Assets/Images/s3.png", width=135)
        s3 = st.checkbox('s3',key='s3_check')
        if s3:
            pass
    with col5:
        st.image("Assets/Images/cosmos.png", width=100)
        cdb = st.checkbox('CosmosDb',key='cdb_check')
        if cdb:
            dataframes_dict = {}
            for container_properties in database.list_containers():
                container_name = container_properties['id']
                container = database.get_container_client(container_name)
                items = list(container.query_items(
                    query="SELECT * FROM c",
                    enable_cross_partition_query=True
                ))
                cleaned_items = [clean_cosmos_item(item) for item in items]
                df = pd.DataFrame(cleaned_items)
                dataframes_dict[container_name] = df
            full_data = full_data | dataframes_dict   
            st.markdown('##')
            st.write('CosmosDb data loaded.')
            st.markdown('##')
            st.metric(label="Cosmos", value=f"{len(dataframes_dict)} Tables", delta=f"{sum(len(df) for df in dataframes_dict.values())} Rows")
    
    st.divider()
    st.subheader("Begin Data Linking.")
    if st.button("Link Data Sources",key='link_data'):
        with st.status("Data linkage started.", expanded=True) as status:
            collections = [x.name for x in qdrant_client.get_collections().collections]
            full_data_unstructured = full_data.pop('unstructured', None)

            for table_name, table in full_data.items():
                table['_full_text'] = table.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
                # table['_md5_hash'] = table['_full_text'].apply(lambda x: hashlib.md5(str(x).encode()).hexdigest())
                table['_embedding'] = table['_full_text'].apply(lambda x: get_embedding(x))
            
                if table_name not in collections:
                    qdrant_client.create_collection(
                        table_name,
                        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                        hnsw_config=HnswConfigDiff(
                            m=32,  
                            ef_construct=400,  
                            full_scan_threshold=16384,  
                        )
                    )  
                    points = [
                        PointStruct(
                            id=index,
                            vector=row['_embedding'],
                            payload={
                                "full_text" : row['_full_text'],
                                }
                        ) for index, row in full_data[table_name].iterrows()
                    ]
                    qdrant_client.upsert(collection_name=table_name, points=points)
                    st.write(f"{table_name} vectorised.")
            collections = [x.name for x in qdrant_client.get_collections().collections]
            for collection in collections:
                for other_collection in collections:
                    if collection != other_collection:
                        # code to fetch best matches from a collection to another collection. 
                        collection_points = qdrant_client.scroll(
                            collection_name=collection,
                            limit=100,
                            with_payload=False,
                            with_vectors=True,
                        )
                        # create a search batch query
                        search_queries = []
                        for collection_point in collection_points[0]:
                            search_request = models.SearchRequest(vector=collection_point.vector, limit=1)
                            search_queries.append(search_request)
                        # Find matching vector from other collection to every point in collection.
                        matching_vectors = qdrant_client.search_batch(collection_name=other_collection, requests=search_queries)
                        # update payload for each point in the collection to best meaches from other collection
                        for collection_point,matching_vector in zip(collection_points[0],matching_vectors):
                            pt_text= qdrant_client.retrieve(
                                collection_name=other_collection,
                                ids=[matching_vector[0].id]
                            )[0]
                            qdrant_client.set_payload(
                                collection_name=collection,
                                payload={
                                    other_collection: {"id":matching_vector[0].id, "score":matching_vector[0].score, "full_text": pt_text.payload['full_text']},
                                },
                                points=[collection_point.id],
                            )
                        st.write(f"{collection} linked to {other_collection}")
            status.update(label="Linking complete!", state="complete", expanded=False)

elif choice == 'Data Simulation':
    st.image("Assets/Images/logo.png", width=200)
    st.title('Insight PII')
    st.subheader('PII identification and management solutions')
    st.markdown('##')
    st.write("Seamlessly unify your organization's data sources, effortlessly identify records for individuals,"
             " and even uncover references within unstructured data, documents, and images. Unleash the full "
             "potential of your data integration and record identification needs today!")
    st.subheader('', divider='rainbow')
    st.markdown('##')

    st.header('Strcutured Data Sources')
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Postgres", "Snowflake", "BigQuery", "Redshift", "Bring Your Data"])

    with tab1:
        st.image("Assets/Images/postgres.png", width=100)
        st.subheader("Postgresql Database")
        uploaded_files = st.file_uploader("Drag and drop CSV files or select files.", type=["csv"],
                                        accept_multiple_files=True, key='pg_upload')
        if uploaded_files:
            if st.button(f"Upload All files", key='pg_raw_upload'):
                with st.status("Uploading data to postgres...", expanded=True) as status:
                    for uploaded_file in uploaded_files:
                        df = pd.read_csv(uploaded_file)
                        df = process_dataframe(df)
                        table_name = uploaded_file.name.replace(' ','_').replace('.csv','').upper()
                        df.to_sql(table_name, engine, schema='insightpii_raw', if_exists='replace', index=False)
                        st.write(f'{table_name} uploaded to Postgres.')
                    status.update(label="Upload complete!", state="complete", expanded=False)
        st.markdown('##')
        postgres_raw_delete = st.toggle('Delete Postgres data', key="postgres_raw")
        if postgres_raw_delete:
            with st.status("Deleting postgresql data...", expanded=True) as status:
                schema_name = 'insightpii_raw'
                query = text(f"SELECT table_name FROM information_schema.tables WHERE table_schema = :schema")
                with engine.connect() as conn:
                    transaction = conn.begin()
                    tables = conn.execute(query, {'schema': schema_name}).fetchall()
                    for table in tables:
                        table_name = table[0]
                        drop_table_query = text(f'DROP TABLE IF EXISTS "{schema_name}"."{table_name}"')
                        conn.execute(drop_table_query)
                        transaction.commit()
                        st.write(f"Table {table_name} dropped.")
                status.update(label="Postgresql db wiped.", state="complete", expanded=False)

    with tab2:
        st.image("Assets/Images/snowflake.png", width=100)
        st.subheader("Snowflake Database")
        uploaded_files = st.file_uploader("Drag and drop CSV files or select files.", type=["csv"],
                                        accept_multiple_files=True, key='sf_upload')
        if uploaded_files:
            if st.button(f"Upload All files", key='sf_raw_upload'):
                with st.status("Uploading data to Snowflake...", expanded=True) as status:
                    for uploaded_file in uploaded_files:
                        df = pd.read_csv(uploaded_file)
                        df = process_dataframe(df)
                        table_name = uploaded_file.name.replace(' ','_').replace('.csv','').upper()
                        upload_to_snowflake(connection=sf_conn, dataframe=df, file=table_name, schema=SNOWFLAKE_RAW_SCHEMA)
                        st.write(f'{table_name} uploaded to Snowflake.')
                    status.update(label="Upload complete!", state="complete", expanded=False)
        st.markdown('##')
        snowflake_raw_delete = st.toggle('Delete Snowflake data', key="sf_raw")
        if snowflake_raw_delete:
            with st.status("Deleting snowflake data...", expanded=True) as status:
                cur = sf_conn.cursor()
                cur.execute(f"""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'RAW_DATA'
                AND table_type = 'BASE TABLE';
                """)
                tables = cur.fetchall()
                for table_name, in tables:
                    st.write(f"Deleting records from table: {table_name}")
                    delete_query = f"DROP TABLE INSIGHTAIQ.RAW_DATA.{table_name}"
                    cur.execute(delete_query)
                    st.write(f"{table_name} dropped.")
                cur.close()
                status.update(label="Snowflake db wiped.", state="complete", expanded=False)

    with tab3:
        st.image("Assets/Images/bigquery.png", width=100)
        st.subheader("Google BigQuery")
        uploaded_files = st.file_uploader("Drag and drop CSV files or select files.", type=["csv"],
                                        accept_multiple_files=True, key='bq_upload')
        st.markdown('##')
        bigquery_raw_delete = st.toggle('Delete Bigquery data', key="bq_raw")

    with tab4:
        st.image("Assets/Images/redshift.png", width=100)
        st.subheader("Amazon Redshift")
        uploaded_files = st.file_uploader("Drag and drop CSV files or select files.", type=["csv"],
                                        accept_multiple_files=True, key='rs_upload')
        st.markdown('##')
        redshift_raw_delete = st.toggle('Delete Redshift data', key="rs_raw")

    with tab5:

        byod = pd.DataFrame({
            'Patient_Name': ['Name 1', 'Name 2', 'Name 3'],
            'Patient_Address': ['Address 1', 'Address 2', 'Address 3'],
            'Patient_emailid':['email1@gmail.com','email2@gmail.com','email3@org1.com'],
            'Diagnosis_code':['code1','code2','code3'],
            'Toal_bill': [1,2,3]
        })
        # if 'original_boyd' not in st.session_state:
        #     st.session_state['original_boyd'] = byod.copy()   
        st.subheader("Bring Your Own Data")
        editable_df = st.data_editor(byod,num_rows="dynamic",hide_index=True)
        editable_df.dropna(axis=0, inplace=True)
        col1, col2 = st.columns(2)
        with col1: 
            upload_option = st.selectbox(
            'Where would you like to upload this data?',
            ('Snowflake', 'Postgres', 'Cosmos'))

        if st.button("Process this data"):
            if upload_option == "Snowflake":
                st.write(upload_option)
                st.dataframe(editable_df)
            
            elif upload_option == "Postgres":
                st.write(upload_option)
                st.dataframe(editable_df)

            elif upload_option == "Cosmos":
                
                st.dataframe(editable_df)
            else:
                st.write(f"red:[Please select the database to persist your data.]")

    st.divider()
    st.header('Unstructured Data Sources')
    tab1, tab2, tab3 = st.tabs(["ADLS", "S3", "Cosmos"])

    with tab1:  
        st.image("Assets/Images/adls.png", width=100)
        st.subheader("Azure Data Lake Storage gen2.")
        image_files = st.file_uploader("Drag and drop files or select files", accept_multiple_files=True, key='azure_sa_upload')
        if image_files:
            if st.button(f"Upload All files", key='azure_sa_upload_btn'):
                with st.status("Uploading data to ADLS...", expanded=True) as status:
                    for image_file in image_files:
                        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
                        blob_client = blob_service_client.get_blob_client(container="adls", blob=image_file)
                        blob_client.upload_blob(image_file, overwrite=True)
                        st.write(f"{image_file.name} uploaded successfully!")
                    status.update(label="Upload complete.", state="complete", expanded=False)
        st.markdown('##')
        azuresa_delete = st.toggle('Delete ADLS data', key="azuresa_del")
        if azuresa_delete:
            with st.status("Deleting Azure Storage data...", expanded=True) as status:
                blob_service_client = BlobServiceClient.from_connection_string(connect_str)
                container_client = blob_service_client.get_container_client('adls')
                blob_list = container_client.list_blobs()
                for blob in blob_list:
                    blob_client = blob_service_client.get_blob_client(container='adls', blob=blob.name)
                    blob_client.delete_blob(delete_snapshots='include')
                    st.write(f'{blob.name} deleted from ADLS.')
                status.update(label="Azure Storage wiped clean", state="complete", expanded=False)

    with tab2: 
        # Remove references of adls and change to s3  
        st.image("Assets/Images/s3.png", width=130)
        st.subheader("AWS S3 buckets")
        image_files = st.file_uploader("Drag and drop files or select files", accept_multiple_files=True, key='adls_upload')
        # if image_files:
        #     if st.button(f"Upload All files", key='adls_sa_upload_btn'):
        #         for image_file in image_files:
        #             blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        #             blob_client = blob_service_client.get_blob_client(container="adlsg2", blob=image_file)
        #             blob_client.upload_blob(image_file, overwrite=True)
        #             st.toast(f"{image_file.name} uploaded successfully!")
        #         st.success("All files uploaded.")
        st.markdown('##')
        adls_delete = st.toggle('Delete S3 data', key="adls_del")
        if adls_delete:
            pass
            # with st.status("Deleting ADLS data...", expanded=True) as status:
            #     blob_service_client = BlobServiceClient.from_connection_string(connect_str)
            #     container_client = blob_service_client.get_container_client('adlsg2')
            #     blob_list = container_client.list_blobs()
            #     for blob in blob_list:
            #         blob_client = blob_service_client.get_blob_client(container='adlsg2', blob=blob.name)
            #         blob_client.delete_blob(delete_snapshots='include')
            #         st.toast(f'{blob.name} deleted from ADLS gen2.')
            #     status.update(label="ADLS gen2 wiped clean", state="complete", expanded=False)

    with tab3:
        st.image("Assets/Images/cosmos.png", width=100)
        st.subheader("Azure CosmosDb")
        image_files = st.file_uploader("Drag and drop files or select files", accept_multiple_files=True, key='cosmos_upload')
        if image_files:
            if st.button(f"Upload All files", key='cosmos_upload_btn'):
                with st.status("Uploading data to CosmosDb...", expanded=True) as status:
                    for image_file in image_files:
                        df = pd.read_csv(image_file)
                        df = process_dataframe(df)
                        df['id'] = df.apply(lambda row: create_hash(row), axis=1)
                        json_data = df.to_json(orient='records')
                        records = json.loads(json_data)
                        container_name = image_file.name.replace(' ','_').replace('.csv','').upper()
                        container = database.create_container_if_not_exists(
                            id=container_name,
                            partition_key=PartitionKey(path=f"/id"),
                            offer_throughput=400
                        )
                        for record in records:
                            try:
                                container.upsert_item(record)
                            except exceptions.CosmosHttpResponseError as e:
                                st.write(f"Error inserting record: {record}")
                                st.write(f"Cosmos DB Error: {e}")
                                break
                        st.write(f'{image_file} uploaded to CosmosDb')
                    status.update(label="All files uploaded to CosmosDb", state="complete", expanded=False)
        st.markdown('##')
        cosmos_delete = st.toggle('Delete Cosmos data', key="cosmos_del")
        if cosmos_delete:
            with st.status("Deleting Cosmos data...", expanded=True) as status:
                try:
                    for container_properties in database.list_containers():
                        container_name = container_properties['id']
                        st.write(f"Deleting container: {container_name}")
                        database.delete_container(container_name)
                except exceptions.CosmosHttpResponseError as e:
                    st.write(f"An error occurred: {e}")
                status.update(label="CosmosDbwiped clean", state="complete", expanded=False)


    st.divider()
    st.header('Vector Databases')
    tab1, tab2, tab3 = st.tabs(["Qdrant", "Pinecone", "Neon"])

    with tab1:
        st.image("Assets/Images/qdrant.jpeg", width=100)
        st.subheader("Qdrant Vector Database")
        qd_delete = st.toggle('Delete qdrant data', key="qdrant")
        if qd_delete:
            collections = qdrant_client.get_collections()
            for x in collections.collections:
                qdrant_client.delete_collection(collection_name=x.name)
                st.toast(f"{x.name} deleted from qdrant.")
            st.success("All vectors removed.")

    with tab2:
        st.image("Assets/Images/pinecone.png", width=100)
        st.subheader("Pinecone Vector Database")
        pc_delete = st.toggle('Delete Pinecone data', key="pcone")

    with tab3:
        st.image("Assets/Images/neon.png", width=100)
        st.subheader("Neon Vector Database")
        neon_delete = st.toggle('Delete Neon data', key="Neon")

elif choice == 'Identify Records':
    st.image("Assets/Images/logo.png", width=200)
    st.title('Insight PII')
    st.subheader('PII identification and management solutions')
    st.markdown('##')
    st.write("Seamlessly unify your organization's data sources, effortlessly identify records for individuals,"
             " and even uncover references within unstructured data, documents, and images. Unleash the full "
             "potential of your data integration and record identification needs today!")
    st.subheader('', divider='rainbow')
    st.markdown('##')

    st.header('Identify an entity across datasets.')
    st.markdown('##')
    title = st.text_input('Type in a few attributes about the entity you want to find. See an example', 'Johonson White 10932 Brigge road jwhite@domain.com')
    st.markdown('##')
    col1, col2, col3 = st.columns(3)
    with col1:
        confidence_score_selected = st.slider('How confident do you want to be?', 80, 100, 85)
    st.markdown('##')
    if st.button("find me"):
        identified_entities = pd.DataFrame(columns=["collection", "id", "score", "text"])
        st.markdown('##')
        st.markdown('##')
        collections = [x.name for x in qdrant_client.get_collections().collections if x.name != 'unstructured']
        for collection in collections:
            response = qdrant_client.search(
                collection_name=collection,
                query_vector=get_embedding(title),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )[0]
            if response:
                entity = pd.DataFrame.from_dict({
                    "collection": [collection],
                    "id": [response.id],
                    "score": [response.score],
                    "text": [response.payload['full_text']]
                })
                identified_entities = pd.concat([identified_entities, entity], ignore_index=True)
        identified_entities.sort_values(by='score',inplace=True, ascending=False)
        st.write(identified_entities)
        new_entities = pd.DataFrame(columns=["collection", "id", "score", "text"])
        for row in identified_entities.itertuples():
            new_point = qdrant_client.retrieve(
                collection_name=getattr(row, 'collection'),
                ids=[getattr(row, 'id')],
            )[0].payload
            new_point.pop('full_text')
            for x in new_point:
                new_entity = pd.DataFrame.from_dict({
                    "collection": [x],
                    "id": [new_point[x]['id']],
                    "score": [new_point[x]['score']],
                    "text": [new_point[x]['full_text']]
                })
                new_entities = pd.concat([new_entities, new_entity], ignore_index=True)
        identified_entities = pd.concat([identified_entities, new_entities], ignore_index=True)
        identified_entities.sort_values(by='score',inplace=True, ascending=False)
        identified_entities.sort_values(by=['collection', 'id', 'text', 'score'], ascending=[True, True, True, False], inplace=True)
        # Step 1: Find the ID with maximum occurrences for each collection
        max_occurrences = identified_entities.groupby(['collection', 'id']).size().reset_index(name='counts')
        max_ids = max_occurrences[max_occurrences.groupby(['collection'])['counts'].transform(max) == max_occurrences['counts']]
        # Step 2: Filter the DataFrame to keep only rows with those IDs
        filtered_df = identified_entities.merge(max_ids[['collection', 'id']], on=['collection', 'id'])
        # Step 3: For each collection, keep only the row with the highest score
        final_df = filtered_df.loc[filtered_df.groupby('collection')['score'].idxmax()]
        st.write(final_df)
 

        # st.write("step1: best matches") ##
        # st.write(best_matches) ##
        # highest_score_collection = max(best_matches, key=lambda k: best_matches[k]['score'])
        # match_text = best_matches[highest_score_collection]["payload"].pop('full_text')
        # entry = pd.DataFrame.from_dict({
        #     "collection": [highest_score_collection],
        #     "id": [best_matches[highest_score_collection]["id"]],
        #     "score":  [best_matches[highest_score_collection]["score"]],
        #     "text": [match_text],
        #     # "payload": [best_matches[highest_score_collection]["payload"]]
        # })
        # identified_entities = pd.concat([identified_entities, entry], ignore_index=True)
        


        # if best_matches[highest_score_collection]['score'] < confidence_score_selected/100:
        #     st.write(f':red[No matches found within selected confidence scores.]')
        #     html = '<h1>IDENTIFIED DATA</h1>'
        # else: 
        #     for collection in collections:
        #         if collection != highest_score_collection:
        #             comparison_vector = best_matches[highest_score_collection]['vector']
        #             response = qdrant_client.search(
        #                 collection_name=collection,
        #                 query_vector=comparison_vector,
        #                 limit=1,
        #                 with_payload=True,
        #                 with_vectors=True,
        #             )[0]
        #             if response:
        #                 best_match = response
        #                 if best_match.score > best_matches[highest_score_collection]['score']:
        #                     highest_score_collection = collection
        #                     best_matches[collection] = {
        #                         "payload": best_match.payload,
        #                         "score": best_match.score,
        #                         "vector": best_match.vector
        #                     }
            
        #     final_comparison_collection = collections[0] if highest_score_collection != collections[0] else collections[1]
        #     comparison_vector = best_matches[highest_score_collection]['vector']

        #     response = qdrant_client.search(
        #         collection_name=final_comparison_collection,
        #         query_vector=comparison_vector,
        #         limit=1,
        #         with_payload=True,
        #         with_vectors=True,
        #     )[0]
        #     if response:
        #         best_match = response
        #         best_matches[final_comparison_collection] = {
        #             "payload": best_match.payload,
        #             "score": best_match.score,
        #             "vector": best_match.vector
        #         }

        #     full_data={}

        #     sf_df = populate_sf_data(sf_conn, SNOWFLAKE_RAW_SCHEMA)
        #     full_data = full_data | sf_df

        #     schema_name = 'insightpii_raw'
        #     pg_query = f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema_name}'"
        #     pg_table_names = pd.read_sql(pg_query, engine)
        #     tables_dict = {}
        #     for table_name in pg_table_names['table_name']:
        #         query = f'SELECT * FROM "{schema_name}"."{table_name}"'
        #         tables_dict[table_name] = pd.read_sql(query, engine)
        #     full_data = full_data | tables_dict

        #     dataframes_dict = {}
        #     for container_properties in database.list_containers():
        #         container_name = container_properties['id']
        #         container = database.get_container_client(container_name)
        #         items = list(container.query_items(
        #             query="SELECT * FROM c",
        #             enable_cross_partition_query=True
        #         ))
        #         cleaned_items = [clean_cosmos_item(item) for item in items]
        #         df = pd.DataFrame(cleaned_items)
        #         dataframes_dict[container_name] = df
        #     full_data = full_data | dataframes_dict

        #     for table in full_data:
        #         cols_to_concatenate = full_data[table].columns.difference(['_source'])
        #         full_data[table]['_full_text'] = full_data[table][cols_to_concatenate].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
            
        #     html = '<h1>IDENTIFIED DATA</h1>'
            
        #     for table, content in best_matches.items():
        #         if content['score'] > confidence_score_selected/100:
        #             st.write(f":green[Found in Database: **{table}**, with confidence of **{content['score']* 100:.2f}%**]")
        #         else:
        #             st.write(f":red[Needs Human Review for: **{table}**, with confidence of **{content['score']* 100:.2f}%**]")
        #         st.dataframe(full_data[table][full_data[table]['_full_text']==content['payload']['full_text']].iloc[:,:-1], hide_index = True)
        #         html+=f"<h3> Table:{table} with confidence {content['score']* 100:.2f}% </h3>"
        #         html+=pd.DataFrame(full_data[table][full_data[table]['_full_text']==content['payload']['full_text']].iloc[:,:-1]).to_html()
        #         html+='<p>----------------------------</p>'
        #         st.divider()

        # us_response = qdrant_client.search(
        #         collection_name='unstructured',
        #         query_vector=get_embedding(title),
        #         limit=10,
        #         with_payload=True,
        #         with_vectors=True,
        #         score_threshold=confidence_score_selected/100,
        #     )
        
        # for resp in us_response:
        #     if resp.score * 100 > confidence_score_selected + 5:
        #         st.write(f":green[Found in document **{resp.payload['source']}**, with confience of **{resp.score * 100:.2f}%**]")
        #         html+=f"<h3>Found in document **{resp.payload['source']}**, with confience of **{resp.score * 100:.2f}%**</h3>"
        #     elif resp.score * 100 > confidence_score_selected + 3:
        #         st.write(f":red[Human review needed for document **{resp.payload['source']}**, with confience of **{resp.score * 100:.2f}%**]")
        #         html+=f"<h3>Human review needed document **{resp.payload['source']}**, with confience of **{resp.score * 100:.2f}%**</h3>"
        #     else:
        #         st.write(f":red[Fringe Match for document **{resp.payload['source']}**, with confience of **{resp.score * 100:.2f}%**]")
        #         html+=f"<h3>Fringe Matched on document **{resp.payload['source']}**, with confience of **{resp.score * 100:.2f}%**</h3>"

        #     account_name = 'insightunstructured'
        #     container_name = 'adls'
        #     blob_name = resp.payload['source']
        #     blob_url = get_sas_token(account_name=account_name, container_name=container_name, blob_name=blob_name)
        #     st.write(f"Download link:  {blob_url}")
        #     if str(blob_name).endswith(".pdf"):
        #         file_name = str(blob_name).split(".pdf")[0]
        #         blob_service_client = BlobServiceClient.from_connection_string(connect_str)           
        #         download_blob_to_file(blob_service_client, container_name, blob_name, f'./Assets/{file_name}.png')
        #         displayPDF(f'./Assets/{file_name}.png')
        #         st.divider()
        #     else:
        #         st.image(blob_url, caption='Found Image', width=300)
        #         st.divider() 
        # st.markdown('##')
        # st.download_button('Download Report', html, file_name='report.html')
        




        



    


