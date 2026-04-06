import { createBrowserRouter } from "react-router";
import { Layout } from "./components/Layout";
import { Home } from "./pages/Home";
import { Processing } from "./pages/Processing";
import { Result } from "./pages/Result";
import { Dashboard } from "./pages/Dashboard";
import { History } from "./pages/History";
import { Profile } from "./pages/Profile";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Layout,
    children: [
      { index: true, Component: Home },
      { path: "dashboard", Component: Dashboard },
      { path: "history", Component: History },
      { path: "profile", Component: Profile },
    ],
  },
  { path: "/processing", Component: Processing },
  { path: "/result", Component: Result },
]);