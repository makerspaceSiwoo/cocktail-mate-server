# 이미지 저장용 S3 버킷(공개 읽기) + 앱 전용 IAM 사용자/액세스 키

# S3 버킷명은 전역 유니크해야 하므로 임의 suffix를 붙인다.
resource "random_id" "suffix" {
  byte_length = 4
}

resource "aws_s3_bucket" "images" {
  bucket = "${var.project_name}-images-${random_id.suffix.hex}"
}

# 이미지를 공개 URL로 서빙하기 위해 퍼블릭 접근 허용.
# presigned URL만 쓸 거라면 이 블록과 아래 bucket_policy를 제거하면 된다.
resource "aws_s3_bucket_public_access_block" "images" {
  bucket                  = aws_s3_bucket.images.id
  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

resource "aws_s3_bucket_policy" "public_read" {
  bucket     = aws_s3_bucket.images.id
  depends_on = [aws_s3_bucket_public_access_block.images]
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "PublicReadObjects"
      Effect    = "Allow"
      Principal = "*"
      Action    = "s3:GetObject"
      Resource  = "${aws_s3_bucket.images.arn}/*"
    }]
  })
}

# 앱(FastAPI)이 S3에 업로드/삭제할 때 쓰는 IAM 사용자.
# 최소 권한: 이 버킷에 대한 객체 R/W/삭제 + 목록.
resource "aws_iam_user" "app" {
  name = "${var.project_name}-app"
}

resource "aws_iam_access_key" "app" {
  user = aws_iam_user.app.name
}

resource "aws_iam_user_policy" "app_s3" {
  name = "${var.project_name}-s3-access"
  user = aws_iam_user.app.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject",
        "s3:ListBucket",
      ]
      Resource = [
        aws_s3_bucket.images.arn,
        "${aws_s3_bucket.images.arn}/*",
      ]
    }]
  })
}
