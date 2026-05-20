import { useState } from "react";
import { HomeScreen } from "./components/HomeScreen";
import { CameraScanScreen } from "./components/CameraScanScreen";
import { ExplanationScreen } from "./components/ExplanationScreen";

type Screen = "home" | "camera" | "explanation";

export default function App() {
  const [currentScreen, setCurrentScreen] =
    useState<Screen>("home");

  const handleStartCamera = () => {
    setCurrentScreen("camera");
  };

  const handleCapture = () => {
    setCurrentScreen("explanation");
  };

  const handleBackToHome = () => {
    setCurrentScreen("home");
  };

  return (
    <div className="size-full bg-gradient-to-br from-gray-100 to-gray-200 flex items-center justify-center p-4">
      <div className="relative w-full max-w-sm h-[90vh] max-h-[800px] bg-black rounded-[3rem] shadow-2xl overflow-hidden">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-32 h-6 bg-black rounded-b-2xl z-20" />

        <div className="relative w-full h-full bg-white overflow-hidden">
          {currentScreen === "home" && (
            <HomeScreen onStartCamera={handleStartCamera} />
          )}
          {currentScreen === "camera" && (
            <CameraScanScreen
              onCapture={handleCapture}
              onBack={handleBackToHome}
            />
          )}
          {currentScreen === "explanation" && (
            <ExplanationScreen onReset={handleBackToHome} />
          )}
        </div>
      </div>
    </div>
  );
}