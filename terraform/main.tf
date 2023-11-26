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

# Azure Container Instance (ACI)
resource "azurerm_container_group" "container" {
  name                = "insightpii-container"
  location            = azurerm_resource_group.insightpii-rg.location
  resource_group_name = azurerm_resource_group.insightpii-rg.name
  os_type             = "Linux"
  
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
      # ... other environment variables ...
    }
  }
  # Assign a public IP address to the container group
  ip_address {
    type = "Public"
    ports {
      port     = 8501
      protocol = "TCP"
    }
  }
}
