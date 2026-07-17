# OCI provider — API 키 인증.
# 인증값은 OCI 콘솔 > 프로필 > API Keys 에서 발급 (DEPLOY.md 2단계 참고).
provider "oci" {
  tenancy_ocid     = var.tenancy_ocid
  user_ocid        = var.user_ocid
  fingerprint      = var.fingerprint
  private_key_path = var.private_key_path
  region           = var.region
}
