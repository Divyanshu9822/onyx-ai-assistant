# Onyx AI Assistant

Onyx AI Assistant is an advanced AI-powered virtual assistant capable of long-term memory and real-time interaction using various tools and technologies. This project provides unique features like GitHub repository management, face recognition, camera-based object detection, screen interaction, and Google Lens-like search, making it a comprehensive AI solution for developers and general users.

> **Note**  
> 
> - The **text-based assistant** is available on the `main` branch.
> - The **speech-based assistant** (supporting voice-to-voice conversation) is available on the `speech-based-assistant` branch.

## Demo

[Loom Video](https://www.loom.com/share/1264867b9c0042f3b5294d2f784d722e?sid=0e645ffb-c3c7-4eec-90ed-a20058b313fc)

## Table of Contents
- [Project Description](#project-description)
- [Features](#features)
- [Installation](#installation)
  - [Clone and Setup](#clone-and-setup)
  - [Environment Setup (Windows, Mac, Linux)](#environment-setup)
- [Technologies Used](#technologies-used)
- [Contact](#contact)

## Project Description

Onyx AI Assistant leverages the power of AI to create an interactive and versatile virtual assistant. It comes equipped with long-term memory, enabling it to recall past interactions and recognize users. Onyx can execute a variety of tasks, such as managing GitHub repositories, analyzing visual input through a camera, answering questions about objects in view, and performing Google Lens searches. This makes Onyx ideal for hands-free assistance with coding, research, and even personal management tasks.

## Features

- **Long-Term Memory**: Uses persistent memory storage to remember interactions over time.
- **Face Recognition**: Recognizes users' faces to personalize responses.
- **Camera Interaction**: Can detect and describe objects in view, answer questions based on visual input, and perform Google Lens-like searches.
- **Screen Analysis**: Identifies elements on the screen and responds to user queries about visible content.
- **GitHub Integration**: Can create and clone repositories, facilitating seamless interaction with GitHub.
  
## Installation

### Clone and Setup

To get started, clone the repository from GitHub:

```bash
git clone https://github.com/Divyanshu9822/onyx-ai-assistant.git
cd onyx-ai-assistant
```

### Environment Setup

The following steps cover setting up the virtual environment on both Windows and Mac/Linux systems.

#### For Windows Users

1. **Using Python's Virtual Environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Using Conda (optional)**
   ```powershell
   conda create -n onyx_env python=3.x
   conda activate onyx_env
   pip install -r requirements.txt
   ```

#### For Linux/Mac Users

1. **Using Python's Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Using Conda (optional)**
   ```bash
   conda create -n onyx_env python=3.x
   conda activate onyx_env
   pip install -r requirements.txt
   ```

### Run the Application

To start the Onyx AI Assistant, run:

```bash
python main.py
```

## Technologies Used

Onyx AI Assistant utilizes a diverse set of tools and libraries:
- **Gemini and OpenAI**: For graph-based memory and general AI processing.
- **[Deepface](https://github.com/serengil/deepface)**: For face recognition and verification.
- **LangChain and LangSmith**: Enabling intelligent conversation and interaction.
- **[Cloudinary](https://cloudinary.com)**: For handling image uploads.
- **[Mem0 AI](https://mem0.ai)**: Provides long-term memory storage for extended AI memory capabilities.


## Contact

For any inquiries or support, please contact [Divyanshu Prasad](https://divyanshuprasad.dev/)

---

Feel free to contribute, create issues, or submit pull requests for improvements!