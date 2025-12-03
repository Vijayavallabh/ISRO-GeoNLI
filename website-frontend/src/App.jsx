import { useState } from 'react'
import './App.css'
import { BrowserRouter as Router, Routes, Route, Link, Navigate } from 'react-router-dom';
import Home from './home';
import Login from './login';
import Tome from './test';

function App() {

  return (
      <Router>
        <Routes>
          <Route
            path="/login"
            element={<Login />}
            //element={isLoggedIn ? <Navigate to="/home" /> : <Login setIsLoggedIn={setIsLoggedIn} />}
          />
          <Route
            path="/test"
            element={<Tome />}
            //element={isLoggedIn ? <Navigate to="/home" /> : <Signup />}
          />
          <Route
            path="/home"
            element={<Home />}
            //element={isLoggedIn ? <Home setIsLoggedIn={setIsLoggedIn} /> : <Navigate to="/login" />}
          />
          <Route path="*" element={<Navigate to="/login" />} />
        </Routes>
      </Router>
  )
}

export default App
