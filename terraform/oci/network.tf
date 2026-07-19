# VCN·게이트웨이·라우트·보안리스트·서브넷 (네트워크 리소스는 모두 무료)

# 가상 클라우드 네트워크
resource "oci_core_vcn" "this" {
  compartment_id = var.compartment_ocid
  cidr_blocks    = ["10.0.0.0/16"]
  display_name   = "cocktail-mate-vcn"
  dns_label      = "cocktailvcn"

  lifecycle {
    prevent_destroy = true
  }
}

# 외부 인터넷 통신용 게이트웨이
resource "oci_core_internet_gateway" "this" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.this.id
  display_name   = "cocktail-mate-igw"
  enabled        = true
}

# 0.0.0.0/0 → 인터넷 게이트웨이 라우트
resource "oci_core_route_table" "this" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.this.id
  display_name   = "cocktail-mate-rt"

  route_rules {
    destination       = "0.0.0.0/0"
    destination_type  = "CIDR_BLOCK"
    network_entity_id = oci_core_internet_gateway.this.id
  }
}

# 인바운드 방화벽: SSH=지정 IP만 / HTTP·HTTPS=공개, 아웃바운드=전체
resource "oci_core_security_list" "this" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.this.id
  display_name   = "cocktail-mate-sl"

  # 아웃바운드 전체 허용
  egress_security_rules {
    destination = "0.0.0.0/0"
    protocol    = "all"
  }

  # SSH(22): 지정 대역(var.ssh_allowed_cidr)만 허용 — 보안
  ingress_security_rules {
    protocol = "6"
    source   = var.ssh_allowed_cidr
    tcp_options {
      min = 22
      max = 22
    }
  }

  # PostgreSQL(5432): 개발 단계라 공개 — 개발자들이 공인 IP로 직접 접속.
  # 방어선은 강한 비밀번호(전 계정 20자+ 랜덤) + 권한 분리(cm_app/cm_dml은 DDL 불가).
  # 실사용자 데이터가 들어오면 이 규칙을 제거하고 compose 바인딩을 127.0.0.1로 되돌린다.
  ingress_security_rules {
    protocol = "6"
    source   = "0.0.0.0/0"
    tcp_options {
      min = 5432
      max = 5432
    }
  }

  # HTTP(80) / HTTPS(443): var.web_ingress_cidrs 대역 허용 (기본 공개,
  # Cloudflare 도입 후엔 CF IP 대역만 남겨 origin 직접 공격 차단)
  dynamic "ingress_security_rules" {
    for_each = {
      for pair in setproduct([80, 443], var.web_ingress_cidrs) :
      "${pair[0]}-${pair[1]}" => pair
    }
    content {
      protocol = "6"
      source   = ingress_security_rules.value[1]
      tcp_options {
        min = ingress_security_rules.value[0]
        max = ingress_security_rules.value[0]
      }
    }
  }
}

# 퍼블릭 서브넷
resource "oci_core_subnet" "public" {
  compartment_id    = var.compartment_ocid
  vcn_id            = oci_core_vcn.this.id
  cidr_block        = "10.0.1.0/24"
  display_name      = "cocktail-mate-public-subnet"
  dns_label         = "public"
  route_table_id    = oci_core_route_table.this.id
  security_list_ids = [oci_core_security_list.this.id]

  prohibit_public_ip_on_vnic = false
}
