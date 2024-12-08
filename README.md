<div align="center">

# AuRA Avatars

The text to video functionality in this repository is based on the LTX-Video project.

[Website](https://www.lightricks.com/ltxv) |
[Repo](https://github.com/Lightricks/LTX-Video) 

</div>

## AuRA Avatar Generator

The **AuRA Avatar Generator** is a companion tool designed to enhance the capabilities of the **AuRA** project and the **IntellectCascade** framework. It allows the creation of highly customizable avatars and avatar-driven video presentations tailored to diverse audiences. This tool serves as a dynamic interface for delivering research insights, improving engagement, and bridging the communication gap in healthcare and other fields.

---

### Features

- **Customizable Avatars**:
  - Personalize avatars based on attributes such as age, gender, race, hairstyle, clothing, accessories, and more.
  - Tailor avatars to reflect the diverse needs of specific audiences.

- **Video Generation**:
  - Generate high-quality avatar-driven videos with customizable resolution, frame rate, and length.
  - Supports dynamic settings for professional lighting, background objects, and scene context.

- **Integration-Ready**:
  - Designed to integrate with the **AuRA** and **IntellectCascade** ecosystems, enabling autonomous generation of avatar presentations.

- **Real-Time Insights**:
  - Generate videos on-demand based on the latest research or dynamically changing parameters.

- **Scalable Output**:
  - Produce single avatar images or complete video sequences to fit various use cases, from patient education to professional conferences.

---

### Technology Stack

- **Frontend**:
  - Gradio-based user interface for easy customization and parameter adjustment.

- **Backend**:
  - Uses machine learning pipelines powered by models such as VAE (Video Autoencoder), Transformer3D, and Rectified Flow Scheduler for realistic video generation.
  - Built on PyTorch, with advanced video processing using tools like MoviePy and Real-ESRGAN for upscaling.

- **Integration Capabilities**:
  - Seamlessly integrates with IntellectCascade's multiagent AI framework for automated and autonomous workflows.

---

### Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/eoffermann/AuRA_Avatars_old.git
   cd AuRA_Avatars_old
   ```

2. **Install Dependencies**:
   Ensure Python 3.8+ is installed, and then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Application**:
   ```bash
   python app.py
   ```
   This will launch the Gradio interface in your browser for avatar customization and video generation.

---

### Usage

- **Customize Avatars**:
  - Use the dropdowns in the Gradio interface to set attributes like age, hair color, clothing type, and background objects.
  - Adjust video settings such as resolution, frame rate, and number of frames.

- **Generate Content**:
  - Click the "Generate Avatars" button to produce your avatar video or image.
  - Outputs are saved to the `outputs` directory by default.

- **Command-Line Usage**:
  - Run the pipeline directly with custom parameters:
    ```bash
    python inference.py --ckpt_dir models/checkpoints --prompt "A professional-looking avatar for a news broadcast"
    ```

---

### Contribution Guidelines

We welcome contributions to improve the AuRA Avatar Generator. To contribute:
- Fork the repository and create a feature branch.
- Make your changes and submit a pull request with a detailed description.

---

### License

This project is licensed under a combination license (parts are MIT License, parts are Apache 2). See the `LICENSE` file for details.

---

### About

The AuRA Avatar Generator is part of the **AuRA** (Autonomous Research Assistant) and **IntellectCascade** ecosystem developed by **Big Blue Ceiling**. It supports evidence-based reporting, enabling healthcare providers and other professionals to deliver insights with greater engagement and accessibility.

For more information, visit:
- **Big Blue Ceiling Official Website**: [big-blue-ceiling.com](https://big-blue-ceiling.com)
- **Documentation**: [IntellectCascade](https://big-blue-ceiling.documentation.com)

---

Feel free to replace placeholder links or add specific sections based on additional project requirements.
