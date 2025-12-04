// import { Form, Row, Col, Input, Button, message as er } from "antd";
// import axios from "axios";
// import { useContext } from "react";
// import { ThemeContext } from "./utils/style";
// import { useNavigate } from "react-router-dom";

// const Login = () => {
//     const navigateUser = useNavigate();
//     const { darkMode, toggleDarkMode } = useContext(ThemeContext);

//     const handleSubmit = async (v) => {
//         try {
//           console.log(v);
//         const res = await axios.post("http://localhost:5000/api/login", v,{
//           headers:{"Content-Type": "application/json"}
//         });
//         if (res) {
// 				er.success("logged in");
// 				console.log(res.data);
// 				localStorage.setItem("token", res.data, {
// 					expiresIn: "1d",
// 				});
// 				navigateUser("/home");
// 			}
//         } catch (e) {
//         console.log(e.message);
//         er.error(e.message);
//         }
//     };

//   return (
//     <>
//       <div
//         style={{
//           height: "70px",
//           backgroundColor: darkMode ? "#05072c" : "#ffffff",
//           padding: "0 2rem",
//           display: "flex",
//           justifyContent: "space-between",
//           alignItems: "center",
//           borderBottom: darkMode ? "1px solid #1d1f3f" : "1px solid #eaeaea",
//           boxShadow: darkMode
//             ? "0 3px 12px rgba(0,0,0,0.45)"
//             : "0 3px 12px rgba(0,0,0,0.10)",
//           overflow: "hidden",
//         }}
//       >
//         <h2 style={{ color: darkMode ? "#ffffff" : "#000000", margin: 0 }}>
//           geoNLI Image Chat
//         </h2>

//         <button
//           onClick={toggleDarkMode}
//           style={{
//             background: darkMode ? "#ffffff" : "#000000",
//             color: darkMode ? "#000000" : "#ffffff",
//             padding: "0.5rem 1.2rem",
//             border: "none",
//             borderRadius: "8px",
//             cursor: "pointer",
//             fontWeight: 600,
//             transition: "0.2s",
//           }}
//         >
//           {darkMode ? "Light" : "Dark"}
//         </button>
//       </div>

//       <div
//         style={{
//           height: "calc(100vh - 80px)",
//           display: "flex",
//           alignItems: "center",
//           justifyContent: "center",
//           backgroundColor: darkMode ? "#05072c" : "#f2f4f7",
//           overflow: "hidden",
//           padding: "1rem",
//         }}
//       >
//         <div
//           style={{
//             width: "380px",
//             padding: "2rem",
//             borderRadius: "16px",
//             background: darkMode ? "#1a1f4a" : "#ffffff",
//             boxShadow: darkMode
//               ? "0 8px 20px rgba(0,0,0,0.5)"
//               : "0 8px 20px rgba(0,0,0,0.15)",
//             border: darkMode ? "1px solid #2e3366" : "1px solid #e3e3e3",
//             transition: "0.3s",
//           }}
//         >
//           <h2
//             style={{
//               textAlign: "center",
//               marginBottom: "1.5rem",
//               color: darkMode ? "#ffffff" : "#000000",
//             }}
//           >
//             Login
//           </h2>

//           <Form onFinish={handleSubmit} layout="vertical">
//             <Form.Item
//               name="email"
//               label={
//                 <span style={{ color: darkMode ? "#ddd" : "#333" }}>Email</span>
//               }
//               rules={[{ required: true, message: "Please enter your Email" }]}
//             >
//               <Input size="large" placeholder="Enter your email" />
//             </Form.Item>

//             <Form.Item
//               name="password"
//               label={
//                 <span style={{ color: darkMode ? "#ddd" : "#333" }}>
//                   Password
//                 </span>
//               }
//               rules={[{ required: true, message: "Please enter your Password" }]}
//             >
//               <Input.Password size="large" placeholder="Enter your password" />
//             </Form.Item>

//             <Button
//               htmlType="submit"
//               size="large"
//               style={{
//                 width: "100%",
//                 marginTop: "1rem",
//                 background: "linear-gradient(90deg, #6c63ff, #4f47f5)",
//                 color: "#ffffff",
//                 border: "none",
//                 borderRadius: "10px",
//                 fontWeight: 600,
//                 letterSpacing: "0.5px",
//                 padding: "0.8rem",
//                 cursor: "pointer",
//                 transition: "0.2s",
//               }}
//             >
//               Login
//             </Button>
//           </Form>
//         </div>
//       </div>
//     </>
//   );
// };

// export default Login;
import { Form, Input, Button, message as er } from "antd";
import axios from "axios";
import { useContext } from "react";
import { ThemeContext } from "./utils/style";
import { useNavigate } from "react-router-dom";

// Import Inter font
//import "./fonts.css"; // Create this file and add @import for Inter from Google Fonts

const Login = () => {
  const navigateUser = useNavigate();
  const { darkMode, toggleDarkMode } = useContext(ThemeContext);

  const handleSubmit = async (v) => {
    try {
      console.log(v);
      const res = await axios.post("http://localhost:5000/api/login", v, {
        headers: { "Content-Type": "application/json" },
      });

      if (res) {
        er.success("Logged in successfully!");
        console.log(res.data);
        sessionStorage.setItem("token", res.data.access_token);
        navigateUser("/home");
      }
    } catch (e) {
      console.log(e.message);
      er.error(e.message);
    }
  };

  return (
    <div
      style={{
        fontFamily: "'Inter', sans-serif",
      }}
    >
      {/* Header */}
      <div
        style={{
          height: "70px",
          backgroundColor: darkMode ? "#0f111a" : "#f7f7f8",
          padding: "0 2rem",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          borderBottom: darkMode ? "1px solid #1f2937" : "1px solid #e5e7eb",
          boxShadow: darkMode
            ? "0 3px 12px rgba(0,0,0,0.45)"
            : "0 3px 12px rgba(0,0,0,0.10)",
        }}
      >
        <div>
          <span style={{ margin: 0, fontSize:"1.3rem",display:"inline", fontWeight:"bold" }}>geoNLI</span>
          <span style={{ margin: 0, fontSize:"1.3rem" }}> Image Chat</span>
        </div>
        <button
          onClick={toggleDarkMode}
          style={{
            background: darkMode ? "#e5e7eb" : "#111827",
            color: darkMode ? "#111827" : "#f7f7f8",
            padding: "0.5rem 1.2rem",
            border: "none",
            borderRadius: "8px",
            cursor: "pointer",
            fontWeight: 600,
            transition: "0.2s",
          }}
        >
          {darkMode ? "Light" : "Dark"}
        </button>
      </div>

      {/* Main Login Container */}
      <div
        style={{
          height: "calc(100vh - 70px)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          backgroundColor: darkMode ? "#0f111a" : "#f7f7f8",
          padding: "1rem",
        }}
      >
        <div
          style={{
            width: "380px",
            padding: "2rem",
            borderRadius: "16px",
            background: darkMode ? "#1a1b2a" : "#ffffff",
            boxShadow: darkMode
              ? "0 8px 20px rgba(0,0,0,0.5)"
              : "0 8px 20px rgba(0,0,0,0.15)",
            border: darkMode ? "1px solid #2c2f3e" : "1px solid #e5e7eb",
            transition: "0.3s",
          }}
        >
          <h2
            style={{
              textAlign: "center",
              marginBottom: "1.5rem",
              color: darkMode ? "#e5e7eb" : "#111827",
            }}
          >
            Login
          </h2>

          <Form onFinish={handleSubmit} layout="vertical">
            <Form.Item
              name="email"
              label={
                <span style={{ color: darkMode ? "#e5e7eb" : "#374151" }}>
                  Email
                </span>
              }
              rules={[{ required: true, message: "Please enter your Email" }]}
            >
              <Input size="large" placeholder="Enter your email" />
            </Form.Item>

            <Form.Item
              name="password"
              label={
                <span style={{ color: darkMode ? "#e5e7eb" : "#374151" }}>
                  Password
                </span>
              }
              rules={[{ required: true, message: "Please enter your Password" }]}
            >
              <Input.Password size="large" placeholder="Enter your password" />
            </Form.Item>

            <Button
              htmlType="submit"
              size="large"
              style={{
                width: "100%",
                marginTop: "1rem",
                background: darkMode
                  ? "linear-gradient(90deg, #4f46e5, #6366f1)"
                  : "linear-gradient(90deg, #10b981, #3b82f6)",
                color: "#ffffff",
                border: "none",
                borderRadius: "10px",
                fontWeight: 600,
                letterSpacing: "0.5px",
                padding: "0.8rem",
                cursor: "pointer",
                transition: "0.2s",
              }}
            >
              Login
            </Button>
          </Form>
        </div>
      </div>
    </div>
  );
};

export default Login;
