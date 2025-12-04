# Website Frontend

React + Vite web interface for the ISRO-GeoNLI remote sensing image analysis platform.

## Overview

A modern, responsive web application that provides:
- User authentication (login/signup)
- Image upload and query interface
- Real-time chat with ISRO-GeoNLI API
- Session and message history management
- Annotated image visualization

**Tech Stack:**
- React 18.2
- Vite 7.2 (build tool)
- React Router 7.9 (navigation)
- Ant Design 6.0 (UI components)
- Axios (HTTP client)

## Architecture

```
website-frontend/
├── src/
│   ├── main.jsx         # Application entry point
│   ├── App.jsx          # Root component with routing
│   ├── login.jsx        # Authentication page
│   ├── home.jsx         # Main chat interface
│   ├── test.jsx         # Testing/demo page
│   ├── assets/          # Images and static files
│   └── utils/
│       └── style.jsx    # Shared styles
├── public/              # Static assets
├── package.json         # Dependencies
└── vite.config.js       # Vite configuration
```

## Prerequisites

- Node.js v20+ (LTS recommended)
- npm or yarn package manager
- Backend server running (website-backend)

## Installation

### Step 1: Navigate to Frontend Directory

```bash
cd website-frontend
```

### Step 2: Install Dependencies

```bash
npm install
```

### Step 3: Configure Environment

Create a `.env` file in the `website-frontend` directory:

```env
VITE_API_URL=http://localhost:8001/api
```

## Running the Application

### Development Mode

```bash
npm run dev
```

- Vite dev server starts at `http://localhost:5173`
- Hot Module Replacement (HMR) enabled
- Instant updates on file save

### Production Build

```bash
npm run build
npm run preview
```

Build output is in `dist/` directory.

## Main Components

**App.jsx** - Root component with React Router

**login.jsx** - User authentication interface

**home.jsx** - Main application interface with image upload and query submission

**test.jsx** - Testing/demo component

## API Integration

The frontend communicates with the backend API at `http://localhost:8001/api`:

**Endpoints:**
- `POST /login` - User authentication
- `POST /signup` - User registration
- `POST /chat` - Submit query with image
- `GET /sessions` - Get chat sessions
- `GET /messages/{session_id}` - Get messages

**Authentication:**
```javascript
const token = localStorage.getItem('token');
axios.post(url, data, {
  headers: { 'Authorization': `Bearer ${token}` }
});
```

## Configuration

### vite.config.js

```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
})
```


## Scripts

```bash
npm run dev      # Start dev server
npm run build    # Production build
npm run preview  # Preview prod build
npm run lint     # Lint code
```

## Dependencies

- `react` (18.2.0) - UI library
- `vite` (7.2.4) - Build tool
- `react-router-dom` (7.9.6) - Navigation
- `antd` (6.0.0) - Component library
- `axios` (1.13.2) - API client

## License

Part of the ISRO-GeoNLI project.
