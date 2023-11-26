variable "tenant_id" {
  description = "Azure Tenant ID"
}

variable "subscription_id" {
  description = "Azure subscription_id"
  type        = string
}

variable "client_id" {
  description = "Azure client ID"
}

variable "client_secret" {
  description = "Azure client_secret"
  type        = string
}

variable "OPENAI_KEY" {
  description = "OpenAI API Key"
  type        = string
}

variable "AZURE_STORAGE_CONNECTION_STRING" {
  description = "AZURE_STORAGE_CONNECTION_STRING"
  type        = string
}

variable "COG_ENDPOINT" {
  description = "COG_ENDPOINT"
  type        = string
}

variable "COG_KEY" {
  description = "COG_KEY"
  type        = string
}

variable "COSMOS_ENDPOINT" {
  description = "COSMOS_ENDPOINT"
  type        = string
}

variable "COSMOS_KEY" {
  description = "COSMOS_KEY"
  type        = string
}

variable "PG_DB" {
  description = "PG_DB"
  type        = string
}

variable "PG_HOST" {
  description = "PG_HOST"
  type        = string
}

variable "PG_PASSWORD" {
  description = "PG_PASSWORD"
  type        = string
}

variable "PG_PORT" {
  description = "PG_PORT"
  type        = string
}

variable "PG_USER_NAME" {
  description = "PG_USER_NAME"
  type        = string
}

variable "QDRANT_ENDPOINT" {
  description = "QDRANT_ENDPOINT"
  type        = string
}

variable "QDRANT_KEY" {
  description = "QDRANT_KEY"
  type        = string
}

variable "SNOWFLAKE_ACCOUNT" {
  description = "SNOWFLAKE_ACCOUNT"
  type        = string
}

variable "SNOWFLAKE_DATABASE" {
  description = "SNOWFLAKE_DATABASE"
  type        = string
}

variable "SNOWFLAKE_LINKED_SCHEMA" {
  description = "SNOWFLAKE_LINKED_SCHEMA"
  type        = string
}

variable "SNOWFLAKE_PASSWORD" {
  description = "SNOWFLAKE_PASSWORD"
  type        = string
}

variable "SNOWFLAKE_RAW_SCHEMA" {
  description = "SNOWFLAKE_RAW_SCHEMA"
  type        = string
}

variable "SNOWFLAKE_ROLE" {
  description = "SNOWFLAKE_ROLE"
  type        = string
}

variable "SNOWFLAKE_USER" {
  description = "SNOWFLAKE_USER"
  type        = string
}

variable "SNOWFLAKE_WAREHOUSE" {
  description = "SNOWFLAKE_WAREHOUSE"
  type        = string
}

variable "VISION_ENDPOINT" {
  description = "VISION_ENDPOINT"
  type        = string
}

variable "VISION_KEY" {
  description = "VISION_KEY"
  type        = string
}

variable "docker_username" {
  description = "Docker Hub Username"
  type        = string
}

variable "docker_password" {
  description = "Docker Hub Password"
  type        = string
}