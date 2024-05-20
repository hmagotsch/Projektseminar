import { StrictMode } from "react";//für Entwicklungsühase nützlich
import { createRoot } from "react-dom/client";
//import "./styles.css";
import App from "./App";//Hauptkomponente die später gerendert wird (aus rohdaten Bild erzeugen)

//root erstellen
const rootElement = document.getElementById("root");
const root = createRoot(rootElement);

root.render(
  <StrictMode>
    <App>
    </App>
  </StrictMode>
);

