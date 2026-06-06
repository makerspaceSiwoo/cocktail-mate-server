# ---- AWS 인증/리전 ----
variable "region" {
  type        = string
  description = "AWS 리전"
  default     = "ap-northeast-2" # 서울
}

variable "aws_profile" {
  type        = string
  description = "사용할 AWS CLI 프로필 (비우면 기본 자격증명 사용)"
  default     = ""
}

variable "project_name" {
  type    = string
  default = "cocktail-mate"
}

# ---- EC2 (Free Tier) ----
# Free Tier: t2.micro 750시간/월 (신규 계정 가입 후 12개월 한정)
variable "instance_type" {
  type    = string
  default = "t2.micro"
}

variable "ubuntu_version" {
  type    = string
  default = "24.04"
}

variable "root_volume_gb" {
  type        = number
  description = "루트 EBS 볼륨(GB). Free Tier 30GB 이하 권장"
  default     = 30
}

# ---- 접속 ----
variable "ssh_public_key" {
  type        = string
  description = "EC2에 등록할 SSH 공개키 (cat ~/.ssh/id_ed25519.pub)"
}

variable "ssh_allowed_cidr" {
  type        = string
  description = "SSH(22) 허용 대역. 보안상 본인 IP/32 권장 (기본은 전체 허용)"
  default     = "0.0.0.0/0"
}
