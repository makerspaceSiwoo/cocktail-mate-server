# AWS 인증은 코드에 넣지 않는다.
#   방법 1) aws configure 로 ~/.aws/credentials 설정 (권장)
#   방법 2) 환경변수 AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
#   특정 프로필을 쓰려면 tfvars의 aws_profile 지정.
provider "aws" {
  region  = var.region
  profile = var.aws_profile != "" ? var.aws_profile : null

  default_tags {
    tags = {
      Project   = var.project_name
      ManagedBy = "terraform"
    }
  }
}
