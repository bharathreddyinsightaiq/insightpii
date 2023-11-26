terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

# Azure Provider Configuration
provider "azurerm" {
  subscription_id = var.subscription_id  
  features {
    key_vault {
      purge_soft_delete_on_destroy    = true
      recover_soft_deleted_key_vaults = true
    }
  }
  client_id       = var.client_id
  client_secret   = var.client_secret
  tenant_id       = var.tenant_id
}

data "azurerm_client_config" "current" {}

# Resource Group
resource "azurerm_resource_group" "insightpii-rg" {
  name     = "insightpii-rg"
  location = "UK South"
}

# Key Vault
resource "azurerm_key_vault" "insightpii-kv" {
  name                = "insightpii-kv"
  location            = azurerm_resource_group.insightpii-rg.location
  resource_group_name = azurerm_resource_group.insightpii-rg.name
  tenant_id           = var.tenant_id
  sku_name            = "standard"
  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id

    key_permissions = [
      "Get", "List", "Create", "Delete", "Recover", "Backup", "Restore"
    ]

    secret_permissions = [
      "Get", "List", "Set", "Delete", "Recover", "Backup", "Restore"
    ]

    storage_permissions = [
      "Get", "List", "Set", "Delete", "Recover", "Backup", "GetSAS"
    ]
  }
}

# Key Vault Secrets
resource "azurerm_key_vault_secret" "openai_key" {
  name         = "OPENAI-KEY"
  value        = var.OPENAI_KEY
  key_vault_id = azurerm_key_vault.insightpii-kv.id
}

resource "azurerm_key_vault_secret" "SNOWFLAKE_ACCOUNT" {
  name         = "SNOWFLAKE-ACCOUNT"
  value        = var.SNOWFLAKE_ACCOUNT
  key_vault_id = azurerm_key_vault.insightpii-kv.id
}

resource "azurerm_key_vault_secret" "SNOWFLAKE_USER" {
  name         = "SNOWFLAKE-USER"
  value        = var.SNOWFLAKE_USER
  key_vault_id = azurerm_key_vault.insightpii-kv.id
}

resource "azurerm_key_vault_secret" "SNOWFLAKE_PASSWORD" {
  name         = "SNOWFLAKE-PASSWORD"
  value        = var.SNOWFLAKE_PASSWORD
  key_vault_id = azurerm_key_vault.insightpii-kv.id
}

resource "azurerm_key_vault_secret" "SNOWFLAKE_WAREHOUSE" {
  name         = "SNOWFLAKE-WAREHOUSE"
  value        = var.SNOWFLAKE_WAREHOUSE
  key_vault_id = azurerm_key_vault.insightpii-kv.id
}

resource "azurerm_key_vault_secret" "SNOWFLAKE_DATABASE" {
  name         = "SNOWFLAKE-DATABASE"
  value        = var.SNOWFLAKE_DATABASE
  key_vault_id = azurerm_key_vault.insightpii-kv.id
}

resource "azurerm_key_vault_secret" "SNOWFLAKE_RAW_SCHEMA" {
  name         = "SNOWFLAKE-RAW-SCHEMA"
  value        = var.SNOWFLAKE_RAW_SCHEMA
  key_vault_id = azurerm_key_vault.insightpii-kv.id
}

resource "azurerm_key_vault_secret" "SNOWFLAKE_LINKED_SCHEMA" {
  name         = "SNOWFLAKE-LINKED-SCHEMA"
  value        = var.SNOWFLAKE_LINKED_SCHEMA
  key_vault_id = azurerm_key_vault.insightpii-kv.id
}

resource "azurerm_key_vault_secret" "SNOWFLAKE_ROLE" {
  name         = "SNOWFLAKE-ROLE"
  value        = var.SNOWFLAKE_ROLE
  key_vault_id = azurerm_key_vault.insightpii-kv.id
}

resource "azurerm_key_vault_secret" "AZURE_STORAGE_CONNECTION_STRING" {
  name         = "AZURE-STORAGE-CONNECTION-STRING"
  value        = var.AZURE_STORAGE_CONNECTION_STRING
  key_vault_id = azurerm_key_vault.insightpii-kv.id
}

resource "azurerm_key_vault_secret" "PG_HOST" {
  name         = "PG-HOST"
  value        = var.PG_HOST
  key_vault_id = azurerm_key_vault.insightpii-kv.id
}

resource "azurerm_key_vault_secret" "PG_PORT" {
  name         = "PG-PORT"
  value        = var.PG_PORT
  key_vault_id = azurerm_key_vault.insightpii-kv.id
}

resource "azurerm_key_vault_secret" "PG_DB" {
  name         = "PG-DB"
  value        = var.PG_DB
  key_vault_id = azurerm_key_vault.insightpii-kv.id
}

resource "azurerm_key_vault_secret" "PG_USER_NAME" {
  name         = "PG-USER-NAME"
  value        = var.PG_USER_NAME
  key_vault_id = azurerm_key_vault.insightpii-kv.id
}

resource "azurerm_key_vault_secret" "PG_PASSWORD" {
  name         = "PG_PASSWORD"
  value        = var.PG_PASSWORD
  key_vault_id = azurerm_key_vault.insightpii-kv.id
}

resource "azurerm_key_vault_secret" "QDRANT_KEY" {
  name         = "QDRANT-KEY"
  value        = var.QDRANT_KEY
  key_vault_id = azurerm_key_vault.insightpii-kv.id
}

resource "azurerm_key_vault_secret" "QDRANT_ENDPOINT" {
  name         = "QDRANT-ENDPOINT"
  value        = var.QDRANT_ENDPOINT
  key_vault_id = azurerm_key_vault.insightpii-kv.id
}

resource "azurerm_key_vault_secret" "COSMOS_ENDPOINT" {
  name         = "COSMOS-ENDPOINT"
  value        = var.COSMOS_ENDPOINT
  key_vault_id = azurerm_key_vault.insightpii-kv.id
}

resource "azurerm_key_vault_secret" "COSMOS_KEY" {
  name         = "COSMOS-KEY"
  value        = var.COSMOS_KEY
  key_vault_id = azurerm_key_vault.insightpii-kv.id
}

resource "azurerm_key_vault_secret" "VISION_KEY" {
  name         = "VISION-KEY"
  value        = var.VISION_KEY
  key_vault_id = azurerm_key_vault.insightpii-kv.id
}

resource "azurerm_key_vault_secret" "VISION_ENDPOINT" {
  name         = "VISION-ENDPOINT"
  value        = var.VISION_ENDPOINT
  key_vault_id = azurerm_key_vault.insightpii-kv.id
}

resource "azurerm_key_vault_secret" "COG_KEY" {
  name         = "COG-KEY"
  value        = var.COG_KEY
  key_vault_id = azurerm_key_vault.insightpii-kv.id
}

resource "azurerm_key_vault_secret" "COG_ENDPOINT" {
  name         = "COG-ENDPOINT"
  value        = var.COG_ENDPOINT
  key_vault_id = azurerm_key_vault.insightpii-kv.id
}


# Azure Container Instance (ACI)
resource "azurerm_container_group" "container" {
  name                = "insightpii-container"
  location            = azurerm_resource_group.insightpii-rg.location
  resource_group_name = azurerm_resource_group.insightpii-rg.name
  os_type             = "Linux"
  ip_address_type     = "Public"
  dns_name_label      = "insightpii-label"

  image_registry_credential {
    server   = "docker.io"
    username = var.docker_username
    password = var.docker_password
  }

  container {
    name   = "insightpii"
    image  = "insightaiq/insightpii:latest"
    cpu    = "1"
    memory = "1.5"

    ports {
      port     = 8501
      protocol = "TCP"
    }

    # Environment variables from Key Vault
    secure_environment_variables = {
      OPENAI_KEY = azurerm_key_vault_secret.openai_key.value
      SNOWFLAKE_ACCOUNT = azurerm_key_vault_secret.SNOWFLAKE_ACCOUNT.value
      SNOWFLAKE_USER = azurerm_key_vault_secret.SNOWFLAKE_USER.value
      SNOWFLAKE_PASSWORD = azurerm_key_vault_secret.SNOWFLAKE_PASSWORD.value
      SNOWFLAKE_WAREHOUSE = azurerm_key_vault_secret.SNOWFLAKE_WAREHOUSE.value
      SNOWFLAKE_DATABASE = azurerm_key_vault_secret.SNOWFLAKE_DATABASE.value
      SNOWFLAKE_RAW_SCHEMA = azurerm_key_vault_secret.SNOWFLAKE_RAW_SCHEMA.value
      SNOWFLAKE_LINKED_SCHEMA = azurerm_key_vault_secret.SNOWFLAKE_LINKED_SCHEMA.value
      SNOWFLAKE_ROLE = azurerm_key_vault_secret.SNOWFLAKE_ROLE.value
      AZURE_STORAGE_CONNECTION_STRING = azurerm_key_vault_secret.AZURE_STORAGE_CONNECTION_STRING.value
      PG_HOST = azurerm_key_vault_secret.PG_HOST.value
      PG_PORT = azurerm_key_vault_secret.PG_PORT.value
      PG_DB = azurerm_key_vault_secret.PG_DB.value
      PG_USER_NAME = azurerm_key_vault_secret.PG_USER_NAME.value
      PG_PASSWORD = azurerm_key_vault_secret.PG_PASSWORD.value
      QDRANT_KEY = azurerm_key_vault_secret.QDRANT_KEY.value
      QDRANT_ENDPOINT = azurerm_key_vault_secret.QDRANT_ENDPOINT.value
      COSMOS_ENDPOINT = azurerm_key_vault_secret.COSMOS_ENDPOINT.value
      COSMOS_KEY = azurerm_key_vault_secret.COSMOS_KEY.value
      VISION_KEY = azurerm_key_vault_secret.VISION_KEY.value
      VISION_ENDPOINT = azurerm_key_vault_secret.VISION_ENDPOINT.value
      COG_KEY = azurerm_key_vault_secret.COG_KEY.value
      COG_ENDPOINT = azurerm_key_vault_secret.COG_ENDPOINT.value
    }
  }
}
