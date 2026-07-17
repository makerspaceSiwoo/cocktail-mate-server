# 이미지 저장용 오브젝트 스토리지(20GB 무료) + S3 호환 키

# 테넌시의 오브젝트 스토리지 네임스페이스
data "oci_objectstorage_namespace" "ns" {
  compartment_id = var.compartment_ocid
}

# 이미지 버킷 (Standard tier = Always Free 20GB, 공개 읽기)
resource "oci_objectstorage_bucket" "images" {
  compartment_id = var.compartment_ocid
  namespace      = data.oci_objectstorage_namespace.ns.namespace
  name           = var.bucket_name
  access_type    = "ObjectRead"
  storage_tier   = "Standard"

  lifecycle {
    prevent_destroy = true
  }
}

# S3 호환 Access/Secret Key — secret 은 생성 시 1회만 노출되며 state 에 저장됨(state 보안 주의)
resource "oci_identity_customer_secret_key" "s3" {
  display_name = "cocktail-mate-s3"
  user_id      = var.user_ocid

  lifecycle {
    prevent_destroy = true
  }
}
