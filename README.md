# 🚀 Flight Delay Predictor API

A high-performance FastAPI backend that uses machine learning to predict flight delays. Built with Python, scikit-learn, and deployed via Docker on Render.

🔗 **Live API**: [https://flight-predictor-model-ro8r.onrender.com](https://flight-predictor-model-ro8r.onrender.com)  
📖 **API Docs**: [https://flight-predictor-model-ro8r.onrender.com/docs](https://flight-predictor-model-ro8r.onrender.com/docs)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![Status](https://img.shields.io/badge/Status-Production-success)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [API Documentation](#api-documentation)
- [Model Details](#model-details)
- [Deployment](#deployment)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

This FastAPI application serves a trained Logistic Regression model that predicts flight delays based on historical patterns. The API accepts flight parameters and returns predictions in real-time with sub-second response times [web:179][web:182].

### Key Capabilities

- **Real-time Predictions**: Sub-100ms response time for predictions
- **RESTful API**: Clean, well-documented endpoints
- **CORS Enabled**: Supports cross-origin requests from web frontends
- **Auto-generated Docs**: Interactive Swagger UI and ReDoc
- **Dockerized**: Consistent deployment across environments
- **Cloud-ready**: Deployed on Render with auto-scaling

## ✨ Features

- 🤖 **ML Model Serving**: Logistic Regression model with 85%+ accuracy
- 🔄 **Automatic Scaling**: Label encoders and feature scalers included
- 📝 **Request Validation**: Pydantic models ensure data integrity
- 🔒 **CORS Protection**: Secure cross-origin resource sharing
- 📊 **Interactive Docs**: Swagger UI at `/docs` endpoint
- 🐳 **Docker Support**: Containerized for consistent deployment
- ⚡ **High Performance**: Async operations with FastAPI
- 🔍 **Error Handling**: Detailed error messages and status codes

## 🛠 Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Framework** | FastAPI | 0.115.0 |
| **Web Server** | Uvicorn | 0.32.0 |
| **ML Library** | scikit-learn | 1.3.2 |
| **Data Processing** | pandas, numpy | 2.2.3, 2.1.3 |
| **Validation** | Pydantic | 2.9.2 |
| **Serialization** | joblib | 1.4.2 |
| **Containerization** | Docker | Latest |
| **Deployment** | Render | Cloud |

## 📁 Project Structure

