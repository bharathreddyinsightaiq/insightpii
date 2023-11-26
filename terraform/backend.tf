terraform {
  backend "azurerm" {
    resource_group_name   = "Insightpii"
    storage_account_name  = "iaqbrksa"
    container_name        = "tfbackend"
    key                   = "terraform.tfstate"
  }
}
