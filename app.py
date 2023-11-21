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
conn = snowflake.connector.connect(
    user=SNOWFLAKE_USER,
    password=SNOWFLAKE_PASSWORD,
    account=SNOWFLAKE_ACCOUNT,
    warehouse=SNOWFLAKE_WAREHOUSE,
    database=SNOWFLAKE_DATABASE,
    schema=SNOWFLAKE_RAW_SCHEMA
)
conn.cursor().execute(f"USE ROLE {SNOWFLAKE_ROLE}")
conn.cursor().execute(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}")
conn.cursor().execute(f"USE DATABASE {SNOWFLAKE_DATABASE}")
conn.cursor().execute(f"USE SCHEMA {SNOWFLAKE_RAW_SCHEMA}")

#Azure SA
connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

#CososDB
COSMOS_ENDPOINT='https://insightnosql.documents.azure.com:443/'
COSMOS_KEY='MvUd6aSQ91pyS2EaMhKK6S994CnOLsLnurIsMKjyBx4ueqVXDBDCNAkSGHeQ9SRLs5kqL90NLl4rACDb31VJOw=='
cosmos_client = CosmosClient(url=COSMOS_ENDPOINT, credential=COSMOS_KEY)
database = cosmos_client.create_database_if_not_exists(id='insightpii')


#azure vision 
vision_key = "c8b23a0f83a549c8a01a900084e4aaa9"
vision_endpoint = "https://insightpii.cognitiveservices.azure.com/"
credentials = CognitiveServicesCredentials(vision_key)
vision_client = ComputerVisionClient(vision_endpoint, credentials)

#azure cognitive services
cog_key = "746d3d4125bc494e9cdeff77e08f2e72"
cog_endpoint = "https://insightpii-cog.cognitiveservices.azure.com/"
text_analytics_client = TextAnalyticsClient(endpoint=cog_endpoint, credential=AzureKeyCredential(cog_key))

def entity_recognition(client, documents):
    try:
        response = client.recognize_entities(documents=documents)
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
        success, nchunks, nrows, _ = write_pandas(conn=conn, df=dataframe, table_name=file, database=SNOWFLAKE_DATABASE, schema=schema, compression="snappy", quote_identifiers=False)
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

# Page setup
st.set_page_config(page_title="InsightAIQ", layout="wide")
pages = ['Link Records', 'Identify Records', 'Delete Records', 'Admin Panel']
choice = st.sidebar.radio("Choose a page:", pages)

# Page 1 content
if choice == 'Link Records':
    st.image("Assets/Images/logo.png", width=400)
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
            sf_df = populate_sf_data(conn, SNOWFLAKE_RAW_SCHEMA)
            full_data = full_data | sf_df
            st.markdown('##')
            st.write('Snowflake data loaded.')
            st.markdown('##')
            st.metric(label="Snowflake", value=f"{len(sf_df)} Tables", delta=f"{sum(len(df) for df in sf_df.values())} Rows")
    with col2:
        st.image("Assets/Images/Postgres.png", width=100)
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
        st.image("Assets/Images/azuresa.png", width=100)
        sa = st.checkbox('Azure Storage',key='sa_check')
        if sa:
            blob_service_client = BlobServiceClient.from_connection_string(connect_str)
            container_client = blob_service_client.get_container_client(container="iaqbrksa")
            extracted_text_dict = {}
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
                                entities_info += f"{entity.text}"
                updated_text = entities_info
                extracted_text_dict[blob.name] = pd.DataFrame({'text': [updated_text]})
            full_data = full_data | extracted_text_dict  
            st.markdown('##') 
            st.write('Azure Storage data loaded.')
            st.markdown('##')
            st.metric(label="AzureSA", value=f"{len(extracted_text_dict)} Tables", delta=f"{sum(len(df) for df in extracted_text_dict.values())} Rows")
    with col4:
        st.image("Assets/Images/adls.png", width=150)
        adls = st.checkbox('ADLS gen2',key='adls_check')
        if adls:
            blob_service_client = BlobServiceClient.from_connection_string(connect_str)
            container_client = blob_service_client.get_container_client(container="adlsg2")
            extracted_text_dict_adls = {}
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
                                entities_info += f"{entity.text}"
                updated_text = entities_info
                extracted_text_dict_adls[blob.name] = pd.DataFrame({'text': [updated_text]})
            full_data = full_data | extracted_text_dict_adls  
            st.markdown('##')
            st.write('ADLS gen2 data loaded.')
            st.markdown('##')
            st.metric(label="ADLS", value=f"{len(extracted_text_dict_adls)} Tables", delta=f"{sum(len(df) for df in extracted_text_dict_adls.values())} Rows")
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
        st.write(full_data)


elif choice == 'Admin Panel':
    st.image("Assets/Images/logo.png", width=400)
    st.title('Insight PII')
    st.subheader('PII identification and management solutions')
    st.markdown('##')
    st.write("Seamlessly unify your organization's data sources, effortlessly identify records for individuals,"
             " and even uncover references within unstructured data, documents, and images. Unleash the full "
             "potential of your data integration and record identification needs today!")
    st.subheader('', divider='rainbow')
    st.markdown('##')

    st.header('Strcutured Data Sources')
    tab1, tab2, tab3, tab4 = st.tabs(["Postgres", "Snowflake", "BigQuery", "Redshift"])

    with tab1:
        st.image("Assets/Images/Postgres.png", width=100)
        st.subheader("Postgresql Database")
        uploaded_files = st.file_uploader("Drag and drop CSV files or select files.", type=["csv"],
                                        accept_multiple_files=True, key='pg_upload')
        if uploaded_files:
            if st.button(f"Upload All files", key='pg_raw_upload'):
                for uploaded_file in uploaded_files:
                    df = pd.read_csv(uploaded_file)
                    df = process_dataframe(df)
                    table_name = uploaded_file.name.replace(' ','_').replace('.csv','').upper()
                    df.to_sql(table_name, engine, schema='insightpii_raw', if_exists='replace', index=False)
                    st.toast(f'{table_name} uploaded to pg raw.')
                st.success("Postgres DB hydrated now.")
        st.markdown('##')
        postgres_raw_delete = st.toggle('Delete Postgres data', key="postgres_raw")
        if postgres_raw_delete:
            with st.status("Deleting postgresql data...", expanded=True) as status:
                schema_name = 'insightpii_raw'
                query = text(f"SELECT table_name FROM information_schema.tables WHERE table_schema = :schema")
                with engine.connect() as conn:
                    tables = conn.execute(query, schema=schema_name).fetchall()
                    for table in tables:
                        table_name = table[0]
                        conn.execute(f'DROP TABLE IF EXISTS {schema_name}."{table_name}";')
                        st.toast(f"Table {table_name} dropped.")
                status.update(label="Postgresql Raw Schema wiped.", state="complete", expanded=False)

    with tab2:
        st.image("Assets/Images/snowflake.png", width=100)
        st.subheader("Snowflake Database")
        uploaded_files = st.file_uploader("Drag and drop CSV files or select files.", type=["csv"],
                                        accept_multiple_files=True, key='sf_upload')
        if uploaded_files:
            if st.button(f"Upload All files", key='sf_raw_upload'):
                for uploaded_file in uploaded_files:
                    df = pd.read_csv(uploaded_file)
                    df = process_dataframe(df)
                    table_name = uploaded_file.name.replace(' ','_').replace('.csv','').upper()
                    upload_to_snowflake(connection=conn, dataframe=df, file=table_name, schema=SNOWFLAKE_RAW_SCHEMA)
                st.success("Snowflake DB hydrated now.")
        st.markdown('##')
        snowflake_raw_delete = st.toggle('Delete Snowflake data', key="sf_raw")
        if snowflake_raw_delete:
            with st.status("Deleting snowflake data...", expanded=True) as status:
                cur = conn.cursor()
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
                cur.close()
                status.update(label="Snowflake Raw Schema wiped.", state="complete", expanded=False)

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


    st.divider()
    st.header('Unstructured Data Sources')
    tab1, tab2, tab3 = st.tabs(["Azure Storage", "ADLS", "Cosmos"])

    with tab1:  
        st.image("Assets/Images/azuresa.png", width=100)
        st.subheader("Azure Storage Containers.")
        image_files = st.file_uploader("Drag and drop files or select files", accept_multiple_files=True, key='azure_sa_upload')
        if image_files:
            if st.button(f"Upload All files", key='azure_sa_upload_btn'):
                for image_file in image_files:
                    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
                    blob_client = blob_service_client.get_blob_client(container="iaqbrksa", blob=image_file)
                    blob_client.upload_blob(image_file, overwrite=True)
                    st.toast(f"{image_file.name} uploaded successfully!")
                st.success("All files uploaded.")
        st.markdown('##')
        azuresa_delete = st.toggle('Delete azuresa data', key="azuresa_del")
        if azuresa_delete:
            with st.status("Deleting Azure Storage data...", expanded=True) as status:
                blob_service_client = BlobServiceClient.from_connection_string(connect_str)
                container_client = blob_service_client.get_container_client('iaqbrksa')
                blob_list = container_client.list_blobs()
                for blob in blob_list:
                    blob_client = blob_service_client.get_blob_client(container='iaqbrksa', blob=blob.name)
                    blob_client.delete_blob(delete_snapshots='include')
                    st.toast(f'{blob.name} deleted from Azure container.')
                status.update(label="Azure Storage wiped clean", state="complete", expanded=False)

    with tab2:  
        st.image("Assets/Images/adls.png", width=130)
        st.subheader("Azure DataLake Storage Gen2")
        image_files = st.file_uploader("Drag and drop files or select files", accept_multiple_files=True, key='adls_upload')
        if image_files:
            if st.button(f"Upload All files", key='adls_sa_upload_btn'):
                for image_file in image_files:
                    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
                    blob_client = blob_service_client.get_blob_client(container="adlsg2", blob=image_file)
                    blob_client.upload_blob(image_file, overwrite=True)
                    st.toast(f"{image_file.name} uploaded successfully!")
                st.success("All files uploaded.")
        st.markdown('##')
        adls_delete = st.toggle('Delete ADLS data', key="adls_del")
        if adls_delete:
            with st.status("Deleting ADLS data...", expanded=True) as status:
                blob_service_client = BlobServiceClient.from_connection_string(connect_str)
                container_client = blob_service_client.get_container_client('adlsg2')
                blob_list = container_client.list_blobs()
                for blob in blob_list:
                    blob_client = blob_service_client.get_blob_client(container='adlsg2', blob=blob.name)
                    blob_client.delete_blob(delete_snapshots='include')
                    st.toast(f'{blob.name} deleted from ADLS gen2.')
                status.update(label="ADLS gen2 wiped clean", state="complete", expanded=False)

    with tab3:
        st.image("Assets/Images/cosmos.png", width=100)
        st.subheader("Azure CosmosDb")
        image_files = st.file_uploader("Drag and drop files or select files", accept_multiple_files=True, key='cosmos_upload')
        if image_files:
            if st.button(f"Upload All files", key='cosmos_upload_btn'):
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
                    st.toast(f'{image_file} uploaded to cosmos')
                st.success("All files uploaded.")
        st.markdown('##')
        cosmos_delete = st.toggle('Delete Cosmos data', key="cosmos_del")
        if cosmos_delete:
            with st.status("Deleting Cosmos data...", expanded=True) as status:
                try:
                    for container_properties in database.list_containers():
                        container_name = container_properties['id']
                        st.toast(f"Deleting container: {container_name}")
                        database.delete_container(container_name)
                except exceptions.CosmosHttpResponseError as e:
                    st.write(f"An error occurred: {e}")
                status.update(label="CosmosDbwiped clean", state="complete", expanded=False)

# from qdrant_client import QdrantClient
# qdrant_client = QdrantClient(
#     url="https://4e285e1c-046b-4c0c-85e5-d4bafe0663cb.us-east4-0.gcp.cloud.qdrant.io:6333", 
#     api_key="GXtaxI8I-rd4AtL3uArs0veLcavr_-Tnec2nOFmLXxjH08tLqDApaA",
# )



    

    
         
    

