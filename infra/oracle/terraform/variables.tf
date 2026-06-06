# ---- OCI 인증 (필수) ----
variable "tenancy_ocid" {
  type        = string
  description = "테넌시 OCID"
}

variable "user_ocid" {
  type        = string
  description = "API 키를 발급한 사용자 OCID"
}

variable "fingerprint" {
  type        = string
  description = "API 키 지문(fingerprint)"
}

variable "private_key_path" {
  type        = string
  description = "API private key(PEM) 파일 경로"
}

variable "region" {
  type        = string
  description = "리전 식별자 (예: ap-chuncheon-1)"
  default     = "ap-chuncheon-1"
}

variable "compartment_ocid" {
  type        = string
  description = "리소스를 생성할 컴파트먼트 OCID (보통 테넌시 루트 OCID 사용 가능)"
}

# ---- 인스턴스 접속 ----
variable "ssh_public_key" {
  type        = string
  description = "인스턴스 ubuntu 사용자에 등록할 SSH 공개키 내용(ssh-rsa ...)"
}

# ---- 컴퓨트 (Always Free ARM Ampere A1) ----
# Always Free 한도: 합산 4 OCPU / 24GB RAM. 기본값은 여유분을 둔 2 OCPU / 12GB.
variable "instance_ocpus" {
  type        = number
  description = "ARM A1 OCPU 수 (Always Free 합산 4 이하)"
  default     = 2
}

variable "instance_memory_gbs" {
  type        = number
  description = "메모리(GB) (Always Free 합산 24 이하)"
  default     = 12
}

variable "ubuntu_version" {
  type        = string
  description = "Ubuntu 버전 (이미지 조회용)"
  default     = "24.04"
}

variable "instance_display_name" {
  type        = string
  default     = "cocktail-mate-server"
}

# ---- 오브젝트 스토리지 ----
variable "bucket_name" {
  type        = string
  description = "이미지 저장용 버킷 이름"
  default     = "cocktail-images"
}
