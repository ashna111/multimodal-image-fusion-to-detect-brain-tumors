## Multimodal Medical Image Fusion To Detect Brain Tumors 
[In progress]

Images are the largest source of data in healthcare and, at the same time, one of the most difficult sources to analyze. Clinicians today must rely largely on medical image analysis performed by overworked radiologists and sometimes analyze scans themselves. Computer vision software based on the latest deep learning algorithms is already enabling the automated analysis to provide accurate results that are delivered immeasurably faster than the manual process can achieve. **Multimodal medical imaging** can provide us with separate yet complementary structure and function information of a patient study and hence has transformed the way we study living bodies. The motivation for multimodal imaging is to obtain a superior exquisite image that will provide accurate and reliable statistics than any single image while retaining the best functions for the snapshots software program for medically testing, diagnosing and curing diseases.

Diagnostic tools include **Computed tomography (CT)** and **Magnetic resonance imaging (MRI)** and thus these are the two modalities that we will consider for Image Fusion Process. 

We aim to approach a three step process:
1. Image Registeration
2. Image Fusion
3. Image Segmentation

***

### Image Registration
Image registration is the process of transforming images into a common coordinate system so corresponding pixels represent homologous biological points. Registration can be used to obtain an anatomically normalized reference frame in which brain regions from different patients can be compared.
#### Landmark-Based Registration 
Image landmark registration is a simple process where a number of points (landmarks) are defined on the same locations in two volumes. The landmarks are then matched by an algorithm, and the volumes are thus registered. The CT scan image is taken as the reference (fixed) image and the MRI scan image is aligned as per the points selected by the user.

1. **Python Notebook** -
Navigate to `python scripts/Image Registration Process.ipynb` and select the CT and MRI Images.
2. **GUI** -
Setup Flask and install dependencies and run:
`python app.py`
Select the appropriate CT and MRI Images. The registered MRI Image gets saved in `static/mri_registered.jpg`

***

### Image Fusion
