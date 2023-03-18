# Bachelor's thesis: Visual object tracking on embedded devices

**Author: Nik Prinčič**

**Mentor: doc. dr. Luka Čehovin Zajec**

**University of Ljubljana, Faculty of Computer and Information Science**

---

## Abstract

```
In this diploma thesis, a state-of-the-art visual tracker is implemented and evaluated on the embedded device Luxonis OAK-1. The tracker STARK is chosen, which belongs to the family of deep neural network-based trackers. More specifically the tracker uses the novel transformer neural network architecture, which is making its way into increasingly more best performing visual trackers. During the implementation process we had to modify the tracker, convert it to the OpenVINO format, which can be used for inference on the embedded device. To run the processing completely on the embedded device the correct pipeline architecture was also developed, which allows for all the processing to be executed on the embedded device and consequently allows autonomous operation of the embedded device. With this level of autonomy, we successfully decoupled tracking performance from the performance of the host system. We evaluated all versions of the tracker on the VOT2021 and VOT2022. With the evaluation we proved that the tracker ported to the embedded device doesn't lose any tracking performance.
```

# Setup environment

Clone repository

```shell
git clone --recurse-submodules https://github.com/nikp00/stark-on-depthai.git
cd stark-on-depthai
```

Init environment

```shell
./setup.sh --make-env-file --make-python-venv
source venv/bin/activate
```

# Running a demo

There are 4 different demo scripts provided. Each is a combination of host or edge mode and camera feed or synthetic video stream . **With synthetic stream the dataset must be placed in the dataset folder, the setup script can be used to link the dataset to the correct folder (only LASOT and GOT-10K datasets are supported).**

Demo scripts:

- edge_cam_test.py
- edge_synthetic_test.py
- host_cam_test.py
- host.py

All teh development and testing was performed on the Luxonis OAK-1

# Development

TODO
