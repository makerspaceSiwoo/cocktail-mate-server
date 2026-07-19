# 원격 state — OCI Object Storage (S3 호환). 값은 커밋하지 않는다.
#   CI/운영: terraform init -backend-config=<렌더된 backend.hcl>   (secret: TERRAFORM_BACKEND_CONFIG)
#   로컬 검증: terraform init -backend=false && terraform validate
# ⚠️ 이 블록을 추가하면 로컬에서 -backend-config 없이 `terraform init` 하면 backend 설정을 물어본다.
#    원격 state 버킷 생성 + `terraform init -migrate-state`는 런북(Category B)에서 수행한다.
terraform {
  backend "s3" {}
}
