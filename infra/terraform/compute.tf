# ARM Ampere A1 (Always Free) 인스턴스 + Ubuntu 이미지 동적 조회 + cloud-init

data "oci_identity_availability_domains" "ads" {
  compartment_id = var.tenancy_ocid
}

# Canonical Ubuntu, 해당 shape(aarch64)에서 부팅 가능한 최신 이미지 조회
data "oci_core_images" "ubuntu" {
  compartment_id           = var.compartment_ocid
  operating_system         = "Canonical Ubuntu"
  operating_system_version = var.ubuntu_version
  shape                    = "VM.Standard.A1.Flex"
  sort_by                  = "TIMECREATED"
  sort_order               = "DESC"
}

resource "oci_core_instance" "server" {
  compartment_id      = var.compartment_ocid
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
  display_name        = var.instance_display_name
  shape               = "VM.Standard.A1.Flex"

  shape_config {
    ocpus         = var.instance_ocpus
    memory_in_gbs = var.instance_memory_gbs
  }

  source_details {
    source_type = "image"
    source_id   = data.oci_core_images.ubuntu.images[0].id
  }

  create_vnic_details {
    subnet_id        = oci_core_subnet.public.id
    assign_public_ip = true
  }

  metadata = {
    ssh_authorized_keys = var.ssh_public_key
    # Docker 설치 + 인스턴스 내부 방화벽(iptables) 개방
    user_data = base64encode(file("${path.module}/cloud-init.yaml"))
  }
}
