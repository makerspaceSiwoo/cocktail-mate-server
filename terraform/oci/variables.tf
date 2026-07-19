# 입력 변수 (실제 값은 terraform.tfvars 에 작성, 커밋 금지)

# 테넌시(계정) OCID
variable "tenancy_ocid" {
  type = string
}

# API 키를 발급한 사용자 OCID
variable "user_ocid" {
  type = string
}

# API 공개키 지문(fingerprint)
variable "fingerprint" {
  type = string
}

# API private key(PEM) 파일 경로 — 키 파일 자체는 절대 커밋 금지
variable "private_key_path" {
  type = string
}

# 리전 식별자
variable "region" {
  type    = string
  default = "ap-chuncheon-1"
}

# 리소스를 생성할 컴파트먼트 OCID (보통 테넌시 루트와 동일)
variable "compartment_ocid" {
  type = string
}

# 인스턴스 ubuntu 사용자에 등록할 SSH 공개키 내용
variable "ssh_public_key" {
  type = string
}

# SSH(22) 접속 허용 대역 — 보안상 본인 IP/32 권장(기본은 전체 허용)
variable "ssh_allowed_cidr" {
  type    = string
  default = "0.0.0.0/0"
}

# HTTP/HTTPS(80,443) 접속 허용 대역 목록.
# Cloudflare 도입 후 https://www.cloudflare.com/ips-v4 대역들로 교체하면
# 프록시를 우회한 origin 직접 공격이 차단된다 (docs/CLOUDFLARE.md 5단계).
variable "web_ingress_cidrs" {
  type    = list(string)
  default = ["0.0.0.0/0"]
}

# [비용안전] ARM A1 OCPU 수 — Always Free 합산 4 이하만 허용
variable "instance_ocpus" {
  type    = number
  default = 1
  validation {
    condition     = var.instance_ocpus >= 1 && var.instance_ocpus <= 4
    error_message = "Always Free 한도 초과 방지: instance_ocpus 는 1~4 이내여야 합니다."
  }
}

# [비용안전] 메모리(GB) — Always Free 합산 24 이하만 허용
variable "instance_memory_gbs" {
  type    = number
  default = 6
  validation {
    condition     = var.instance_memory_gbs >= 1 && var.instance_memory_gbs <= 24
    error_message = "Always Free 한도 초과 방지: instance_memory_gbs 는 1~24 이내여야 합니다."
  }
}

# [비용안전] 부트 볼륨(GB) — Always Free 200GB 한도 내만 허용
variable "boot_volume_gbs" {
  type    = number
  default = 50
  validation {
    condition     = var.boot_volume_gbs >= 50 && var.boot_volume_gbs <= 200
    error_message = "Always Free 한도 초과 방지: boot_volume_gbs 는 50~200 이내여야 합니다."
  }
}

# 이미지 조회용 Ubuntu 버전
variable "ubuntu_version" {
  type    = string
  default = "24.04"
}

# 인스턴스 표시 이름
variable "instance_display_name" {
  type    = string
  default = "cocktail-mate-server"
}

# 이미지 저장용 버킷 이름
variable "bucket_name" {
  type    = string
  default = "cocktail-images"
}
