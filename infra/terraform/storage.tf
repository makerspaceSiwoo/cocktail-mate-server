# 이미지 저장용 오브젝트 스토리지 버킷 + S3 호환 접근 키

data "oci_objectstorage_namespace" "ns" {
  compartment_id = var.compartment_ocid
}

resource "oci_objectstorage_bucket" "images" {
  compartment_id = var.compartment_ocid
  namespace      = data.oci_objectstorage_namespace.ns.namespace
  name           = var.bucket_name

  # 이미지를 공개 URL로 서빙(익명 읽기 허용). presigned만 쓰려면 NoPublicAccess로 변경.
  access_type = "ObjectRead"
}

# S3 호환 API용 Access/Secret Key.
# 주의: secret(key)은 생성 시 1회만 노출되며 Terraform state에 저장된다 → state를 안전하게 보관할 것.
resource "oci_identity_customer_secret_key" "s3" {
  display_name = "cocktail-mate-s3"
  user_id      = var.user_ocid
}
