# terraform apply 후 출력값을 서버 .env에 옮겨 적는다.

output "instance_public_ip" {
  description = "EC2 공인 IP"
  value       = aws_instance.server.public_ip
}

output "ssh_command" {
  value = "ssh ubuntu@${aws_instance.server.public_ip}"
}

# .env STORAGE_BUCKET
output "s3_bucket" {
  value = aws_s3_bucket.images.bucket
}

# .env STORAGE_ENDPOINT
output "s3_endpoint" {
  value = "https://s3.${var.region}.amazonaws.com"
}

# .env STORAGE_PUBLIC_BASE_URL
output "s3_public_base_url" {
  value = "https://${aws_s3_bucket.images.bucket}.s3.${var.region}.amazonaws.com"
}

# .env STORAGE_ACCESS_KEY
output "s3_access_key" {
  description = "IAM 액세스 키 ID"
  value       = aws_iam_access_key.app.id
  sensitive   = true
}

# .env STORAGE_SECRET_KEY (생성 시에만 확인 가능)
output "s3_secret_key" {
  description = "IAM 시크릿 액세스 키"
  value       = aws_iam_access_key.app.secret
  sensitive   = true
}
