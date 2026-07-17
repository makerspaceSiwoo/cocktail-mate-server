# ARM Ampere A1 인스턴스 + Ubuntu 이미지 조회 + cloud-init (모두 Always Free 범위)

# 인스턴스를 배치할 가용성 도메인 목록
data "oci_identity_availability_domains" "ads" {
  compartment_id = var.tenancy_ocid
}

# 해당 shape(A1/aarch64)에서 부팅 가능한 최신 Ubuntu 이미지
data "oci_core_images" "ubuntu" {
  compartment_id           = var.compartment_ocid
  operating_system         = "Canonical Ubuntu"
  operating_system_version = var.ubuntu_version
  shape                    = "VM.Standard.A1.Flex"
  sort_by                  = "TIMECREATED"
  sort_order               = "DESC"
}

# Always Free ARM 인스턴스 (OCPU/메모리/부트볼륨 모두 무료 한도 내, ephemeral 공인 IP)
resource "oci_core_instance" "server" {
  compartment_id      = var.compartment_ocid
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
  display_name        = var.instance_display_name
  shape               = "VM.Standard.A1.Flex"

  # OCPU/메모리 (variables.tf 에서 Always Free 한도로 검증됨)
  shape_config {
    ocpus         = var.instance_ocpus
    memory_in_gbs = var.instance_memory_gbs
  }

  # 부팅 이미지 + 부트 볼륨 크기(Always Free 200GB 한도 내)
  source_details {
    source_type             = "image"
    source_id               = data.oci_core_images.ubuntu.images[0].id
    boot_volume_size_in_gbs = var.boot_volume_gbs
  }

  # 퍼블릭 서브넷 + ephemeral 공인 IP(무료) 할당
  create_vnic_details {
    subnet_id        = oci_core_subnet.public.id
    assign_public_ip = true
  }

  # SSH 공개키 등록 + cloud-init(Docker 설치, iptables 개방)
  metadata = {
    ssh_authorized_keys = var.ssh_public_key
    user_data           = base64encode(file("${path.module}/cloud-init.yaml"))
  }

  lifecycle {
    prevent_destroy = true
  }
}
