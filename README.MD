# SignSpeak

A real-time sign language recognition and text-to-speech system that converts sign language gestures into spoken audio with emotion-based voice synthesis. Built for presentations and accessibility.

## Features

- **Real-time Sign Language Recognition**: Recognizes both static and dynamic sign language gestures using MediaPipe hand tracking
- **Emotion-Based Text-to-Speech**: Converts recognized text to speech with emotion-aware voice synthesis using Google Cloud Text-to-Speech (Gemini-TTS)
- **Facial Emotion Detection**: Detects facial emotions in real-time using MediaPipe Face Landmarker to enhance TTS with appropriate emotional tone
- **Stream Cleaning**: Intelligent text cleaning pipeline that removes duplicates and trailing noise from recognized gestures
- **Presentation Mode**: Upload and display PDF presentations while signing
- **Speaker Profiles**: Customize voice settings, rate, pitch, and volume
- **Live Captions**: Real-time word-by-word caption display as you sign

## Architecture

### Backend (`backend/`)
- **FastAPI** WebSocket server for real-time communication
- **Gesture Recognition**: SVM classifier for static signs, DTW (Dynamic Time Warping) for dynamic signs
- **Stream Cleaner**: Removes consecutive duplicates, near-duplicates, and trailing noise
- **TTS Service**: Google Cloud Text-to-Speech with Gemini-TTS models for emotion-based synthesis

### Frontend (`frontend/`)
- **React + Vite** application
- **MediaPipe** for hand tracking and facial emotion detection
- **WebSocket** client for real-time backend communication
- **PDF.js** for presentation viewing

## Prerequisites

- **Python 3.8+**
- **Node.js 18+** and npm
- **Google Cloud Account** with Text-to-Speech API enabled
- **Webcam** for sign language recognition and emotion detection
- **Chrome** browser (recommended for best webcam support)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Cheesehacks2026
```

### 2. Backend Setup

#### Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

#### Google Cloud Credentials Setup

1. **Create a Google Cloud Project** (if you don't have one):
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one

2. **Enable Text-to-Speech API**:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Cloud Text-to-Speech API"
   - Click "Enable"

3. **Create Service Account**:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "Service Account"
   - Fill in the service account details
   - Grant the "Cloud Text-to-Speech API User" role
   - Click "Done"

4. **Download Service Account Key**:
   - Click on the created service account
   - Go to "Keys" tab
   - Click "Add Key" > "Create new key"
   - Choose "JSON" format
   - Download the JSON file

5. **Place Credentials File**:
   - Rename the downloaded JSON file to `signspeak-488902-b29067f64881.json`
   - Place it in the project root directory (same level as `backend/` and `frontend/`)
   - Alternatively, set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable:
     ```bash
     export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/credentials.json"
     ```

#### Run the Backend Server

```bash
cd backend
python main.py
```

The backend server will start on **http://localhost:8000**

### 3. Frontend Setup

#### Install Dependencies

```bash
cd frontend
npm install
```

#### Run the Frontend Development Server

```bash
npm run dev
```

The frontend will start on **http://localhost:5173**

### 4. Access the Application

1. Open your browser and navigate to **http://localhost:5173**
2. Allow camera access when prompted
3. The application should connect to the backend automatically (check the connection status in the header)

## Project Structure

```
Cheesehacks2026/
├── backend/
│   ├── main.py              # FastAPI WebSocket server
│   ├── recognizer.py        # Gesture recognition (SVM + DTW)
│   ├── cleaner.py           # Text stream cleaning pipeline
│   ├── features.py          # Feature extraction for gestures
│   ├── tts_service.py       # Google Cloud TTS service
│   ├── templates.json       # Saved sign templates
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main application component
│   │   ├── components/      # React components
│   │   │   ├── Calibration.jsx
│   │   │   ├── DashboardHome.jsx
│   │   │   ├── FacialEmotionDetection.jsx
│   │   │   ├── PresentationsPage.jsx
│   │   │   └── SpeakerProfilesPage.jsx
│   │   └── context/         # React contexts
│   │       └── FacialEmotionContext.jsx
│   ├── package.json
│   └── vite.config.js
├── emotion-app/             # Standalone emotion detection app
├── signspeak-488902-b29067f64881.json  # Google Cloud credentials (not in git)
└── README.md               # This file
```

## Key Technologies

### Backend
- **FastAPI**: Modern Python web framework
- **MediaPipe**: Hand tracking and gesture recognition
- **scikit-learn**: Machine learning (SVM classifier)
- **Google Cloud Text-to-Speech**: Emotion-based voice synthesis
- **NumPy**: Numerical computations
- **WebSockets**: Real-time bidirectional communication

### Frontend
- **React 19**: UI framework
- **Vite**: Build tool and dev server
- **MediaPipe Tasks Vision**: Client-side hand and face tracking
- **PDF.js**: PDF rendering for presentations

## Usage

### Recording Signs

1. **Static Signs** (single gesture):
   - Click "Saved Signs" button
   - Enter a name for your sign
   - Click "Record Static"
   - Perform the gesture and hold it for a moment
   - The system will record multiple samples for better recognition

2. **Dynamic Signs** (motion-based):
   - Enter a name for your sign
   - Click "Record Dynamic"
   - Perform the motion gesture
   - Click "Stop Recording" when done

### Presenting

1. Click the **"Present"** tab
2. Load a PDF presentation (optional) using "Load PDF"
3. Start signing - your gestures will be recognized in real-time
4. Press **E** key to flush the current sentence and trigger TTS
5. The system will:
   - Clean the recognized text (remove duplicates, noise)
   - Detect your facial emotion
   - Synthesize speech with appropriate emotional tone
   - Play the audio through your speakers

### Calibration

1. Go to the **"Calibration"** tab
2. Follow the on-screen instructions
3. The system will calibrate facial emotion detection for your face

## Configuration

### Backend Tuning Parameters

In `backend/main.py`:
- `VOTE_WINDOW`: Number of frames for gesture voting (default: 10)
- `VOTE_THRESHOLD`: Agreement threshold for recognition (default: 0.60)

### Audio Format

The system uses **LINEAR16** (uncompressed WAV) format for highest audio quality. This can be changed in `backend/tts_service.py`:
- `LINEAR16`: Best quality, larger files
- `OGG_OPUS`: Good quality, smaller files
- `MP3`: Standard quality, smaller files

### Emotion Detection

Emotion thresholds can be customized in `frontend/src/components/FacialEmotionDetection.jsx`:
- `STABILITY_FRAMES`: Frames required before emotion change (default: 10)
- Individual emotion thresholds for: confident, warm, excited, emphatic, serious, passionate, reflective

## Troubleshooting

### Backend Issues

**TTS Service Not Initializing**:
- Verify `GOOGLE_APPLICATION_CREDENTIALS` environment variable is set
- Check that the credentials file exists and is valid
- Ensure Text-to-Speech API is enabled in Google Cloud Console

**WebSocket Connection Failed**:
- Verify backend is running on port 8000
- Check firewall settings
- Ensure CORS is properly configured

### Frontend Issues

**Camera Not Working**:
- Use Chrome browser (best support)
- Ensure camera permissions are granted
- Check that no other application is using the camera

**WebSocket Disconnected**:
- Verify backend server is running
- Check browser console for connection errors
- Ensure backend URL in `App.jsx` matches your backend server

### Audio Playback Issues

**No Audio Playing**:
- Check browser audio permissions
- Verify TTS service is initialized in backend
- Check browser console for audio errors
- Ensure audio format is supported by your browser

## Development

### Running in Development Mode

**Backend**:
```bash
cd backend
python main.py
```

**Frontend**:
```bash
cd frontend
npm run dev
```

### Building for Production

**Frontend**:
```bash
cd frontend
npm run build
```

The production build will be in `frontend/dist/`

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Acknowledgments

- MediaPipe for hand and face tracking
- Google Cloud Text-to-Speech for voice synthesis
- FastAPI for the backend framework
- React and Vite for the frontend
