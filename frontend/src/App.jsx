import { useState, useEffect, useRef } from "react";
import "./App.css";

function App() {
  const [detections, setDetections] = useState([]);
  const videoRef = useRef(null);

  useEffect(() => {
    const setupCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("cam:", err);
      }
    };
    setupCamera();

    let ws = null;
    let animationFrameId = null;

    const connectWebSocket = () => {
      try {
        ws = new WebSocket("ws://localhost:8000/ws");

        ws.onopen = () => {
          console.log("connected!");
          sendFrame();
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.recognition) {
              console.log(data.recognition);
            }
          } catch (error) {
            console.error("err parsing socket msgs:", error);
          }
        };

        ws.onerror = (error) => {
          console.error("socket err:", error);
        };

        ws.onclose = (e) => {
          console.log("closed:", e.reason);
          setTimeout(connectWebSocket, 1000);
        };
      } catch (error) {
        console.error("err when connecting socket:", error);
        setTimeout(connectWebSocket, 1000);
      }
    };

    const sendFrame = () => {
      if (ws?.readyState === WebSocket.OPEN && videoRef.current) {
        const canvas = document.createElement("canvas");
        canvas.width = 300;
        canvas.height = 400;
        const ctx = canvas.getContext("2d");

        const videoAspect =
          videoRef.current.videoWidth / videoRef.current.videoHeight;
        let drawWidth = canvas.width;
        let drawHeight = canvas.height;
        let offsetX = 0;
        let offsetY = 0;

        if (videoAspect > 3 / 4) {
          drawWidth = (canvas.height * 3) / 4;
          offsetX = (canvas.width - drawWidth) / 2;
        } else {
          drawHeight = (canvas.width * 4) / 3;
          offsetY = (canvas.height - drawHeight) / 2;
        }

        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(
          videoRef.current,
          offsetX,
          offsetY,
          drawWidth,
          drawHeight
        );

        canvas.toBlob(
          (blob) => {
            if (blob) {
              ws.send(blob);
            }
          },
          "image/jpeg",
          0.8
        );
      }
      animationFrameId = setTimeout(sendFrame, 5000);
    };

    connectWebSocket();

    return () => {
      if (animationFrameId) {
        clearTimeout(animationFrameId);
      }
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach((track) => track.stop());
      }
      if (ws) {
        ws.close();
      }
    };
  }, []);

  return (
    <div>
      <div style={{ position: "relative" }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          style={{ maxWidth: "100%" }}
        />
      </div>
    </div>
  );
}

export default App;
