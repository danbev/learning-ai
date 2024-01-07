### Externa GPU (eGPU)
This document contains note about setting up and configuring an external GPU
to be used with my laptop which is a Lenovo Thinkpad P1 Gen 3.

My main motivation for getting an external GPU is to be able to run CUDA
programs on my laptop and I'm not really interested in using it for the
montitor at this point.

### Hardware
The external GPU enclosure I am using is a Sonnet Technologies, Inc. eGPU
Breakaway Box 750ex, and the GPU is MSI GEFORCE RTX 4070 VENTUS 3X E 12G OC.

### OS/Tools
My laptop is running Fedora 39:
```console
x86_64
Fedora release 39 (Thirty Nine)
NAME="Fedora Linux"
VERSION="39 (Workstation Edition)"
ID=fedora
VERSION_ID=39
VERSION_CODENAME=""
PLATFORM_ID="platform:f39"
PRETTY_NAME="Fedora Linux 39 (Workstation Edition)"
ANSI_COLOR="0;38;2;60;110;180"
LOGO=fedora-logo-icon
CPE_NAME="cpe:/o:fedoraproject:fedora:39"
DEFAULT_HOSTNAME="fedora"
HOME_URL="https://fedoraproject.org/"
DOCUMENTATION_URL="https://docs.fedoraproject.org/en-US/fedora/f39/system-administrators-guide/"
SUPPORT_URL="https://ask.fedoraproject.org/"
BUG_REPORT_URL="https://bugzilla.redhat.com/"
REDHAT_BUGZILLA_PRODUCT="Fedora"
REDHAT_BUGZILLA_PRODUCT_VERSION=39
REDHAT_SUPPORT_PRODUCT="Fedora"
REDHAT_SUPPORT_PRODUCT_VERSION=39
SUPPORT_END=2024-11-12
VARIANT="Workstation Edition"
VARIANT_ID=workstation
Fedora release 39 (Thirty Nine)
Fedora release 39 (Thirty Nine)
```

This used `boltd` which is a thunderbolt daemon that is used to managing
Thunderbolt 3 connections. This will be used when we connect the eGPU to the
laptop using a thunderbolt cable. Thunderbolt 3 is a high-speed I/O interface
that can handle data, video, and power over a single cable and is commonly used
for eGPUs, high-speed storage devices, and docking stations.

we can use `boltctl` to list the connected devices:
```console
$ boltctl list
 ○ Sonnet Technologies, Inc. eGPU Breakaway Box 750ex
   ├─ type:          peripheral
   ├─ name:          eGPU Breakaway Box 750ex
   ├─ vendor:        Sonnet Technologies, Inc.
   ├─ uuid:          000b89c5-f4f8-0800-ffff-ffffffffffff
   ├─ generation:    Thunderbolt 3
   ├─ status:        disconnected
   ├─ authorized:    Sat 06 Jan 2024 10:22:24 AM UTC
   ├─ connected:     Sat 06 Jan 2024 10:22:23 AM UTC
   └─ stored:        Sat 06 Jan 2024 10:22:24 AM UTC
      ├─ policy:     auto
      └─ key:        no
```

### Installation
The installation process was pretty straight forward. The enclosure came as a
single slot but there were two power cables and I was not sure at first which
one to use but they looked pretty simliar so I just picked one and it worked.
I had to push down the cables inside the enclosure to be able to fit the card
in. 
After the card was in I connected the enclosure to my laptop using the provided
thunderbolt cable. I then connected the enclosure to power and turned it on.
Now, at this point my current display which is an externa monitor just froze
which I took as a bad sign and my system crashed but, after thinking about it
a bit more it might have been that connecting the enclosure to my laptop it
took over as the main display and that I should connect the monitor to the
graphics card instead. TODO: try this out.


### Installing and configurating CUDA

Install the CUDA toolkit using the following command:
```console
$ sudo dnf -y install cuda-toolkit-12-3
```

