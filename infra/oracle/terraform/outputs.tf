# terraform apply 후 출력값을 서버 .env에 옮겨 적는다 (DEPLOY.md 4~5단계).

output "instance_public_ip" {
  description = "SSH/HTTP 접속용 공인 IP"
  value       = oci_core_instance.server.public_ip
}

output "ssh_command" {
  description = "인스턴스 접속 명령"
  value       = "ssh ubuntu@${oci_core_instance.server.public_ip}"
}

output "object_storage_namespace" {
  value = data.oci_objectstorage_namespace.ns.namespace
}

output "bucket_name" {
  value = oci_objectstorage_bucket.images.name
}

# .env STORAGE_ENDPOINT 값
output "s3_endpoint" {
  value = "https://${data.oci_objectstorage_namespace.ns.namespace}.compat.objectstorage.${var.region}.oraclecloud.com"
}

# .env STORAGE_PUBLIC_BASE_URL 값
output "s3_public_base_url" {
  value = "https://${data.oci_objectstorage_namespace.ns.namespace}.compat.objectstorage.${var.region}.oraclecloud.com/${oci_objectstorage_bucket.images.name}"
}

# .env STORAGE_ACCESS_KEY 값
output "s3_access_key" {
  description = "Customer Secret Key의 Access Key(ID)"
  value       = oci_identity_customer_secret_key.s3.id
  sensitive   = true
}

# .env STORAGE_SECRET_KEY 값 (생성 시 1회만 확인 가능)
output "s3_secret_key" {
  description = "Customer Secret Key의 Secret"
  value       = oci_identity_customer_secret_key.s3.key
  sensitive   = true
}
