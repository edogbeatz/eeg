# EEG Frontend

A modern Next.js frontend for real-time EEG data visualization and brain signal classification.

## 🚀 Features

- **Real-time EEG Visualization**: 8-channel electrode grid with activity indicators
- **Synthetic Data Integration**: Uses BrainFlow's synthetic board for testing
- **LaBraM Predictions**: Brain signal classification with confidence scores
- **Electrode Status Monitoring**: Connection quality and impedance visualization
- **Modern UI**: Built with Next.js, TypeScript, Tailwind CSS, and Shadcn UI

## 🛠️ Tech Stack

- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS v4
- **UI Components**: Shadcn UI + Radix UI
- **Charts**: Recharts (for future waveform visualization)
- **Icons**: Lucide React

## 🏃‍♂️ Getting Started

### Prerequisites

Make sure the FastAPI backend is running on `http://localhost:8000`

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

Visit `http://localhost:3000` to view the application.

## 🎯 Usage

1. **Start the Backend**: Ensure your FastAPI server is running with the synthetic board endpoints
2. **Open the Frontend**: Navigate to `http://localhost:3000`
3. **Configure Settings**: 
   - Set API URL (default: `http://localhost:8000`)
   - Select data source (Synthetic Board for testing)
   - Enable auto-refresh for continuous updates
4. **Start Streaming**: Click "Start Stream" to begin data collection
5. **View Results**:
   - Channel grid shows real-time activity levels
   - Electrode status displays connection quality
   - Prediction panel shows LaBraM classification results

## 📊 Components

### EEGDashboard
Main dashboard component with controls and data display

### EEGChannelGrid  
8-channel electrode visualization with activity indicators and signal statistics

### ElectrodeStatus
Connection quality monitoring with visual status indicators

### PredictionDisplay
LaBraM classification results with confidence scores and probability breakdown

## 🔧 API Integration

The frontend connects to these backend endpoints:

- `GET /synthetic-data` - Get synthetic EEG data (8 channels × 1000 samples)
- `GET /synthetic-stream` - Get streaming synthetic data (250 samples)
- `POST /predict-synthetic` - Run LaBraM prediction on synthetic data
- `GET /health` - Check API health status

## 🎨 UI Features

- **Responsive Design**: Mobile-first approach with Tailwind CSS
- **Dark Mode Ready**: Built-in dark mode support
- **Real-time Updates**: Auto-refresh capability for continuous monitoring  
- **Visual Feedback**: Color-coded status indicators and progress bars
- **Modern Components**: Shadcn UI components with consistent design

## 🔮 Future Enhancements

- **WebSocket Integration**: Real-time streaming without polling
- **Waveform Visualization**: Interactive EEG signal plots
- **Real Board Support**: Integration with actual OpenBCI Cyton boards
- **Data Export**: Save and export EEG sessions
- **Advanced Analytics**: Signal processing and frequency analysis

## 🤝 Development

```bash
# Install dependencies
npm install

# Start development server with Turbopack
npm run dev

# Build for production
npm run build

# Start production server
npm run start

# Run linting
npm run lint
```

## 📁 Project Structure

```
frontend/
├── src/
│   ├── app/                 # Next.js App Router
│   │   ├── globals.css     # Global styles
│   │   ├── layout.tsx      # Root layout
│   │   └── page.tsx        # Home page
│   ├── components/         # React components
│   │   ├── ui/            # Shadcn UI components
│   │   ├── eeg-dashboard.tsx
│   │   ├── eeg-channel-grid.tsx
│   │   ├── electrode-status.tsx
│   │   └── prediction-display.tsx
│   └── lib/
│       └── utils.ts        # Utility functions
├── public/                 # Static assets
├── components.json         # Shadcn UI config
└── package.json
```