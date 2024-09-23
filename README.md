### Dental X-Ray Segmentation with U-Net and VGG16

This project implements a deep learning solution for **segmentation of dental X-ray images** using the **U-Net architecture** with a **VGG16** encoder as the backbone. The purpose of the model is to accurately segment and classify various dental conditions, such as cavities, crowns, implants, etc., from X-ray images.

#### Key Components:

1. **Data Loading and Preprocessing**:
   - **JSON Annotations Parsing**: The annotation data, stored in COCO format, is loaded and parsed to extract image IDs, file names, and categories. The categories represent different dental conditions.
   - **Image and Mask Loading**: Images and corresponding masks are read from directories. Masks are grayscale images representing different categories for segmentation. Images and masks are resized and normalized to prepare them for model input.
   - **Visualization**: A utility function is provided to visualize a batch of images alongside their ground-truth masks, superimposing the mask over the image for easier interpretation.

2. **Model Architecture - U-Net with VGG16**:
   - **VGG16 Encoder**: The pre-trained VGG16 model from Keras is used as the encoder, with pre-trained weights on ImageNet. The lower layers of VGG16 are frozen to preserve the pre-learned features.
   - **U-Net Decoder**: The U-Net architecture is completed by adding a symmetric decoder, which progressively upsamples the feature maps from VGG16 using transposed convolutions. Skip connections from corresponding encoder layers are concatenated with the decoder to capture both high- and low-level features.
   - **Output Layer**: The model ends with a single-channel convolution layer with a sigmoid activation function, which predicts the segmentation mask for each pixel.

3. **Dataset Generator**:
   - A custom data generator is implemented to load images and their corresponding masks in batches, resizing them to a standard size (256x256) for training. This allows efficient training on large datasets.

4. **Training**:
   - The model is compiled with the **Adam optimizer** and uses **binary cross-entropy** as the loss function, as the task is pixel-wise binary classification (segmenting objects from the background).
   - The model is trained for multiple epochs, with metrics such as **accuracy** being monitored. The dataset is split into training and validation sets to track model performance.
   - The training process outputs a graph showing the evolution of accuracy and loss over time for both training and validation sets.

5. **Evaluation**:
   - After training, the model is evaluated on a test set. Performance metrics such as **F1-Score**, **ROC AUC**, and a confusion matrix are calculated to assess the model's ability to correctly segment the dental conditions.
   - The ROC curve is plotted to visualize the trade-off between true positives and false positives.

6. **Visualization of Results**:
   - A function is provided to visualize the predicted masks over the test images, allowing the user to visually assess the quality of the segmentation.
   
7. **Saving the Model**:
   - Once training is complete, the model is saved in HDF5 format for future use or fine-tuning.



### Key Libraries Used:
- **TensorFlow/Keras**: For defining the U-Net architecture and managing the training process.
- **OpenCV**: For image processing, including loading images and masks.
- **Matplotlib**: For visualizing images, masks, and training metrics.
- **Pandas**: For managing and displaying category and image data.

