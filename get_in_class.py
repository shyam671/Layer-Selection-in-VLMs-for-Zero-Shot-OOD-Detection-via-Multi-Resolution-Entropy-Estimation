def get_in_class_text(args):

    if args.text_prompt == 'single' and args.dataset_dir == 'xray':
        in_class = ['x-ray of normal', 'x-ray of pneumonia']

    if args.text_prompt == 'single' and ('oasis' in args.dataset_dir):
        in_class = ["a healthy brain MRI"," a brain MRI of an healthy adult"]
        #in_class = ["healthy brain, normal anatomy, no stroke lesions, gliomas, or tumors"," brain MRI, healthy adult, no abnormalities"]

    if args.text_prompt == 'multi' and ('oasis' in args.dataset_dir):
        in_class = ["healthy brain, normal anatomy, no stroke lesions, gliomas, or tumors", "axial T1 scan, normal anatomy, no stroke lesions or tumors", "sagittal view, normal structure, no abnormalities", "coronal view, clear ventricles and cortex, no lesions", "high-resolution slice, normal neuroanatomy, no tumors", "3D reconstruction, healthy brain, no gliomas", "normal brain MRI, clear gray and white matter, no lesions", "healthy adult brain, standard MRI scan, no abnormalities", "axial slice, normal brain structures, no stroke or tumors", "sagittal brain MRI, no lesions or gliomas, clear anatomy", "coronal MRI, normal ventricles, no stroke damage", "T1-weighted brain scan, healthy, no tumors", "high-res brain slice, normal neuroanatomy, no lesions", "3D brain MRI, healthy adult, no abnormalities", "axial brain scan, normal anatomy, no gliomas", "sagittal view, normal cortical structure, no lesions", "coronal slice, clear brain ventricles, no tumors", "high-resolution MRI, normal brain, no stroke or gliomas"]

    if args.text_prompt == 'multi-mod' and ('oasis' in args.dataset_dir):
        in_class = ["a healthy brain MRI", "a brain MRI of an healthy adult", "an axial scan of healthy brain", "an axial MRI scan"]
   
    if args.text_prompt == 'multi' and args.dataset_dir == 'xray':
        in_class = ["normal", "healthy", "Pneumonia is indicated in this chest X-ray image.", "This radiograph of the chest shows characteristics typical of pneumonia.", "Signs of pneumonia are evident in the X-ray of the chest.", "The chest radiograph reveals pulmonary patterns consistent with pneumonia.", "Evidence of pneumonia is present in this thoracic X-ray.", "This X-ray of the chest displays diagnostic features of pneumonia.", "Manifestations of pneumonia appear in this chest X-ray.", "Pneumonia is diagnosed in the displayed chest X-ray.", "The chest X-ray examination suggests the presence of pneumonia.", "Chest X-ray analysis indicates pneumonia in this image."]
 
    if args.text_prompt == 'multi-mod' and ('xray' in args.dataset_dir):
        in_class = ["x-ray of normal", "pneumonia indicated in chest X-ray", "chest X-ray shows signs of pneumonia"]

    if args.text_prompt == 'single' and ('midog' in args.dataset_dir):
        in_class = ['a histopathology image of breast carcinoma', 'a histopathology image patch of carcinoma']
        
    if args.text_prompt == 'multi' and ('midog' in args.dataset_dir):
        in_class = ["breast carcinoma histology image", "invasive breast carcinoma slide", "H&E image of breast carcinoma", "microscopic breast carcinoma tissue", "high-resolution breast carcinoma histology", "breast carcinoma micrograph", "zoomed breast carcinoma cells", "breast carcinoma gland structures", "malignant breast epithelial cells", "high magnification breast carcinoma", "invasive ductal breast carcinoma image", "breast carcinoma tumor clusters", "breast carcinoma pathology slide", "breast carcinoma cell detail", "digital breast carcinoma patch", "clinical breast carcinoma image", "malignant breast tissue image", "breast carcinoma H&E section", "invasive breast carcinoma field", "breast carcinoma diagnostic image", "breast carcinoma tissue scan", "breast carcinoma whole-slide image", "breast carcinoma under microscope", "breast carcinoma fibrous stroma", "breast carcinoma tumor margin", "breast carcinoma high power view", "breast carcinoma duct formation", "breast carcinoma mitotic figures", "breast carcinoma nuclear atypia", "breast carcinoma stromal reaction", "breast carcinoma cell clusters", "breast carcinoma architecture distortion", "breast carcinoma abnormal glands", "breast carcinoma tissue section", "breast carcinoma microscopic view", "breast carcinoma pathology image", "breast carcinoma cellular detail", "breast carcinoma zoomed view", "breast carcinoma lab slide", "breast carcinoma tumor cells"]     
    
    if args.text_prompt == 'multi-mod' and ('midog' in args.dataset_dir):
        in_class = ["a good histopathology image of breast carcinoma", "a diagnostic histopathology slide showing breast carcinoma", "a whole-slide style histopathology image of invasive breast carcinoma", "a high quality microscopic histology image of breast carcinoma tissue", "a high-resolution H&E stained histopathology image of breast carcinoma", "a publication-quality histology micrograph of breast carcinoma tissue", "a high quality zoomed in histology image patch showing breast carcinoma", "a zoomed-in histopathology patch of breast carcinoma cells", "a cropped microscopy tile showing malignant breast epithelial cells", "a high magnification histology patch with carcinoma morphology"]

    return in_class