import { RouterProvider } from "react-router";
import { router } from "./routes";
import { SolveProvider } from "./context/SolveContext";

export default function App() {
  return (
    <SolveProvider>
      <RouterProvider router={router} />
    </SolveProvider>
  );
}