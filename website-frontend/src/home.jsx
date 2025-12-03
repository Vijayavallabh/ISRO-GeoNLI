// import { ThemeContext } from "./utils/style";
// import { useContext, useState, useEffect } from "react";
// import { Row, Col, Upload, Button, Input, Select, message as er, InputNumber, Switch } from "antd";
// import { UploadOutlined, SendOutlined, PlusOutlined, MenuOutlined } from "@ant-design/icons";
// import axios from "axios";
// import { v4 as uuidv4 } from "uuid";

// const { Option } = Select;

// const Home = () => {
//   const { darkMode, toggleDarkMode } = useContext(ThemeContext);

//   const [imageUrl, setImageUrl] = useState(null);
//   const [input, setInput] = useState("");
//   const [messages, setMessages] = useState(() => {
//     const stored = sessionStorage.getItem("storedMessages");
//     return stored ? JSON.parse(stored) : [];
//   });
//   const [queryType, setQueryType] = useState("Select query type");
//   const [resolution, setResolution] = useState(null);
//   const [sessions, setSessions] = useState([]);
//   const [newImg, setNewImg] = useState(false);
//   const [sidebar, setSidebar] = useState(false);

//   const generateSessionId = () => {
//     if(sessionStorage.getItem("currentSessionId")){
//       return;
//     }
//     const sessionId = uuidv4();
//     sessionStorage.setItem("currentSessionId", sessionId);
//   }

//   const generateNewChat = () => {
//     setImageUrl(null);
//     setInput("");
//     setMessages([]);
//     setQueryType("Select query type");
//     setResolution(null);
//     sessionStorage.removeItem("currentSessionId");
//     sessionStorage.removeItem("uploadedImage");
//   }

//   const generateSessionByImageUpload = () => {
//     //setImageUrl(null);
//     setInput("");
//     sessionStorage.removeItem("currentSessionId");
//     //sessionStorage.removeItem("uploadedImage");
//     generateSessionId();
//   }

//   const handleUpload = ({ file, onSuccess }) => {
//     const reader = new FileReader();
//     reader.onload = () => {
//       setImageUrl(reader.result);
//       setNewImg(true);
//       sessionStorage.setItem("uploadedImage",reader.result);
//       generateSessionByImageUpload();
//       onSuccess("ok");
//     };
//     reader.readAsDataURL(file);
//   };

//   const handleSend = async () => {
//     const token = sessionStorage.getItem("token");
//     if(!token){
//       er.error("Authentication failed.")
//       return;
//     }
//     if(!sessionStorage.getItem("currentSessionId")){
//       er.error("No session id found.");
//       return;
//     }
//     if (!queryType || queryType==="Select query type") {
//       er.error("Please select a query type.");
//       return;
//     }
//     if (!imageUrl) {
//       er.error("Please upload an image.");
//       return;
//     }
//     if (!resolution) {
//       er.error("Please enter the resolution.");
//       return;
//     }

//     try {
//       // Add user message
//       setMessages(prev => [...prev, { role: "user", content: input }]);
//       setInput("");

//       // Prepare form data
//       const formData = new FormData();
//       formData.append("session_id", sessionStorage.getItem("currentSessionId"));
//       formData.append("query", input);
//       formData.append("query_type", queryType);
//       formData.append("spatial_resolution_m", parseFloat(resolution));
//       if(newImg){
//         const blob = await (await fetch(imageUrl)).blob();
//         formData.append("image", blob, "uploaded_image.png");
//       }
//       else{
//         formData.append("image_url",imageUrl);
//       }

//       // Post to backend
//       const res = await axios.post("http://localhost:5000/api/chat", formData, {
//         headers: { "Content-Type": "multipart/form-data", "Authorization" : `Bearer ${token}` },
//       });

//       setMessages(prev => [...prev, { role: "assistant", content: res.data.reply }]);
//       setNewImg(false);
//       console.log(res.data.response);
//     } catch (e) {
//       console.log(e.message);
//       er.error(e.message);
//     }
//   };

//   const getChatHistory = async () => {
//     try{
//       const token = sessionStorage.getItem("token");
//       console.log(token);
//       const res = await axios.get("http://localhost:5000/api/chat-history",{
//         headers:{"Authorization":`Bearer ${token}`}
//       });
//       console.log(res.data);
//       setSessions(res.data);
//     }
//     catch(e){
//       console.log(e.message);
//       er.error(e.message);
//     }
//   }

//   const handleHistory = async (sessionId) => {
//     console.log(sessionId);
//     try{
//       const token = sessionStorage.getItem("token");
//       sessionStorage.removeItem("currentSessionId");
//       sessionStorage.setItem("currentSessionId", sessionId);
//       const res = await axios.get(`http://localhost:5000/api/get-history`, {
//         headers: { "Authorization": `Bearer ${token}` },
//         params: { sessionId: sessionId }
//       });
//       setImageUrl(res.data.image.image_url);
//       setMessages(res.data.messages)
//       console.log(res.data);
//     }
//     catch(e){
//       er.error(e.message);
//     }
//   }

//   useEffect(() => {
//     sessionStorage.setItem("storedMessages", JSON.stringify(messages));
//   }, [messages]);

//   useEffect(() => {
//     if (imageUrl) sessionStorage.setItem("uploadedImage", imageUrl);
//   }, [imageUrl]);

//   useEffect(() => {
//     const storedImage = sessionStorage.getItem("uploadedImage");
//     if (storedImage) setImageUrl(storedImage);
//     getChatHistory();
//   }, []);

//   const [height, setHeight] = useState(window.innerWidth >= 992 ? "85vh" : "35vh");
//   useEffect(() => {
//     const handleResize = () => setHeight(window.innerWidth >= 992 ? "85vh" : "35vh");
//     window.addEventListener("resize", handleResize);
//     return () => window.removeEventListener("resize", handleResize);
//   }, []);

//   const cardStyle = {
//     background: darkMode ? "#1a1b2a" : "#ffffff",
//     height,
//     borderRadius: "14px",
//     border: darkMode ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.08)",
//     boxShadow: darkMode
//       ? "0 8px 20px rgba(0,0,0,0.5)"
//       : "0 8px 20px rgba(0,0,0,0.15)",
//     transition: "0.3s ease",
//     backdropFilter: "blur(6px)",
//   };

//   return (
//     <div style={{ fontFamily: "'Inter', sans-serif", height:"100vh",backgroundColor: darkMode ? "#0f111a" : "#f7f7f8"}}>
//       {/* HEADER */}
//       <div
//         style={{
//           backgroundColor: darkMode ? "#0f111a" : "#f7f7f8",
//           padding: "0.5rem 1rem",
//           display: "flex",
//           justifyContent: "space-between",
//           alignItems: "center",
//           borderBottom: darkMode ? "1px solid #1f2937" : "1px solid #e5e7eb",
//           boxShadow: darkMode
//             ? "0 2px 8px rgba(0,0,0,0.4)"
//             : "0 2px 8px rgba(0,0,0,0.1)",
//         }}
//       >
//         <div style={{ display: "flex", alignItems: "center", gap: "1rem", color: darkMode ? "#fff" : "#111827" }}>
//           {/* <div style={{ backgroundColor: "#4549b8", padding: "0.5rem", borderRadius: "8px", color: "#fff" }}>
//             <h3>geoNLI</h3>
//           </div> */}
//           <div>
//             <h2 style={{ margin: 0 }}>geoNLI Image Chat</h2>
//             {/* <p style={{ fontSize: "0.65rem", margin: 0 }}>
//               Describe and reason about satellite imagery with a conversational interface.
//             </p> */}
//           </div>
//         </div>
//         <Switch
//           style={{backgroundColor:"#4549b8"}}
//           checked={darkMode}
//           onChange={toggleDarkMode}
//           // checkedChildren="Dark"
//           // unCheckedChildren="Light"
//         />

//       </div>

//       {/* MAIN CONTENT */}
//       <div
//         style={{
//           padding: "15px",
//           backgroundColor: darkMode ? "#0f111a" : "#f7f7f8",
//           transition: "0.3s ease"
//         }}
//       >
//         <Row gutter={20}>
//           {/* LEFT PANEL */}
//           <Col xs={24} lg={sidebar?6:1}>
//             {sidebar?
//               <div style={{ ...cardStyle, height: "85vh", display: "flex", flexDirection: "column", position: "relative" }}>
//               <div style={{display:"flex", justifyContent:"space-between", alignItems:"center", padding:"0.5rem 1rem 0.5rem 1rem",borderBottom: `1px solid ${darkMode ? "#2f335c" : "#d6d6d6"}`, color: darkMode ? "#c7c7c7" : "#111827",}}>
//                 <h4>New chat</h4>
//                 <div>
//                   <Button 
//                     onClick={generateNewChat}
//                     style={{
//                       height: "36px",
//                       padding: "0 15px",
//                       borderRadius: "8px",
//                       background: "#4549b8",
//                       border: "none",
//                       fontWeight: 600,
//                       color:"#fff"
//                     }}
//                     icon={<PlusOutlined />}
//                   ></Button>
//                   <Button
//                     onClick={() => setSidebar(!sidebar)}

//                     style={{
//                         height: "36px",
//                         padding: "0px 5px",
//                         margin:"0.5rem 0 0 1rem",
//                         borderRadius: "8px",
//                         //background: darkMode ? "#1a1b2a" : "#ffffff",
//                         background: "#4549b8",
//                         //color: darkMode ? "#c7c7c7" : "#111827",
//                         border: "none",
//                         fontWeight: 600,
//                         color:"#fff"
//                       }}
//                     icon={<MenuOutlined />}
//                   ></Button>
//               </div>
//               </div>
//               <div style={{ padding: "1rem", color: darkMode ? "#c7c7c7" : "#111827", display:"flex", flexDirection:"row", justifyContent:"space-between", alignItems:"center" }}>
//                 <h4>Chat history</h4>
//               </div>
//               <div style={{ flex: 1, overflowY: "auto", padding: "0.5rem", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
//                 {sessions.map((session) => (
//                   <div key={session.sessionId} style={{ display: "flex", justifyContent:"start"}}>
//                     <div style={{
//                         width: "100%",
//                         padding: "10px 14px",
//                         borderRadius: "15px",
//                         background: darkMode ? "#2f335c" : "#e5e5e5",
//                         color: darkMode ? "#fff" : "#000",
//                         wordWrap: "break-word",
//                         whiteSpace: "pre-wrap",
//                         cursor:"pointer",
//                       }}
//                       onClick={() => handleHistory(session.sessionId)}
//                       onMouseEnter={(e) => (e.currentTarget.style.opacity = 0.8)}
//                       onMouseLeave={(e) => (e.currentTarget.style.opacity = 1)}
//                     >
//                       {session.title}
//                     </div>
//                   </div>
//                 ))}
//               </div>
//             </div>
//             :<div style={{ ...cardStyle, height: "85vh", display: "flex", flexDirection: "column", position: "relative", alignItems:"center" }}>
//               <Button
//                 onClick={() => setSidebar(!sidebar)}
//                 style={{
//                     height: "36px",
//                     padding: "0px 5px",
//                     margin:"0.5rem 0 0 0",
//                     borderRadius: "15px",
//                     //background: darkMode ? "#1a1b2a" : "#ffffff",
//                     background: "#4549b8",
//                     //color: darkMode ? "#c7c7c7" : "#111827",
//                     border: "none",
//                     fontWeight: 600,
//                     color:"#fff"
//                   }}
//                 icon={<MenuOutlined />}
//               ></Button>
//             </div>}
            
//           </Col>


//           <Col xs={24} lg={sidebar?6:11}>
//             <div style={{ ...cardStyle, height: "85vh", display: "flex", flexDirection: "column", position: "relative" }}>

             
//               <div style={{ flex: 1, overflowY: "auto", padding: "0.5rem", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
//                 {messages.map((msg, idx) => (
//                   <div key={idx} style={{ display: "flex", justifyContent: msg.role === "user" ? "flex-end" : "flex-start" }}>
//                     <div style={{
//                       maxWidth: "70%",
//                       padding: "10px 14px",
//                       borderRadius: "15px",
//                       background: msg.role === "user" ? "#4549b8" : darkMode ? "#2f335c" : "#e5e5e5",
//                       color: msg.role === "user" ? "#fff" : darkMode ? "#fff" : "#000",
//                       wordWrap: "break-word",
//                       whiteSpace: "pre-wrap"
//                     }}>
//                       {msg.content}
//                     </div>
//                   </div>
//                 ))}
//               </div>

//               {/* <div style={{display:"flex", width:"100%", justifyContent:"center", alignItems:"center", position:"absolute", bottom:0}}>
//                 <div
//                   style={{position: "absolute",
//                   bottom: "5rem",
//                   width: "100%",
//                   padding:"0.5rem",
//                   boxSizing: "border-box",
//                   background: darkMode ? "#0f112e" : "#f9f9f9",
//                 }}
//                 >
//                   <Select
//                 placeholder="Select query type"
//                 value={queryType}
//                 onChange={setQueryType}
//                 style={{
//                   width: "100%",
//                   margin: "0 0",
//                   background: darkMode ? "#0f112e" : "#fafafa",
//                   color: darkMode ? "#c7c7c7" : "#111827",
//                   border: darkMode ? "1px solid #2b2e55" : "1px solid #ddd",
//                 }}
//               >
//                 <Option value="caption_query">Caption query</Option>
//                 <Option value="grounding_query">Grounding query</Option>
//                 <Option value="attribute_binary_query">Attribute binary query</Option>
//                 <Option value="attribute_numeric_query">Attribute numeric query</Option>
//                 <Option value="attribute_semantic_query">Attribute semantic query</Option>
//               </Select>
//                 </div>
//               </div> */}

              

              

//               <div style={{
//                 display: "flex",
//                 padding: "10px",
//                 height:"3rem",
//                 borderTop: darkMode ? "1px solid #2b2e55" : "1px solid #ddd",
//                 gap: "10px",
//                 position: "absolute",
//                 bottom: "5rem",
//                 width: "100%",
//                 boxSizing: "border-box",
//                 background: darkMode ? "#1a1f4a" : "#f9f9f9",
//                 borderBottomLeftRadius:"15px",
//                 borderBottomRightRadius:"15px"
//               }}>
//                 <Select
//                 placeholder="Select query type"
//                 value={queryType}
//                 onChange={setQueryType}
//                 style={{
//                   width: "100%",
//                   margin: "0 0",
//                   background: darkMode ? "#0f112e" : "#fafafa",
//                   color: darkMode ? "#c7c7c7" : "#111827",
//                   border: darkMode ? "1px solid #2b2e55" : "1px solid #ddd",
//                 }}
//               >
//                 <Option value="caption_query">Caption query</Option>
//                 <Option value="grounding_query">Grounding query</Option>
//                 <Option value="attribute_binary_query">Attribute binary query</Option>
//                 <Option value="attribute_numeric_query">Attribute numeric query</Option>
//                 <Option value="attribute_semantic_query">Attribute semantic query</Option>
//               </Select>
//               </div>
//               <div style={{
//                 display: "flex",
//                 padding: "10px",
//                 borderTop: darkMode ? "1px solid #2b2e55" : "1px solid #ddd",
//                 gap: "10px",
//                 position: "absolute",
//                 bottom: "0rem",
//                 width: "100%",
//                 boxSizing: "border-box",
//                 background: darkMode ? "#1a1f4a" : "#f9f9f9"
//               }}>
//                 <Input
//                   placeholder="Ask something..."
//                   value={input}
//                   onChange={(e) => setInput(e.target.value)}
//                   style={{
//                     flex: 1,
//                     height: "36px",
//                     borderRadius: "15px",
//                     border: darkMode ? "1px solid #2b2e55" : "1px solid #ccc",
//                     background: darkMode ? "#0f112e" : "#fff",
//                     color: darkMode ? "#fff" : "#000",
//                   }}
//                 />
//                 <Button
//                   onClick={handleSend}
//                   style={{
//                     height: "36px",
//                     padding: "0 25px",
//                     borderRadius: "15px",
//                     background: "#4549b8",
//                     border: "none",
//                     fontWeight: 600,
//                     color:"#fff"
//                   }}
//                   icon={<SendOutlined />}
//                 >
//                 </Button>
//               </div>

//             </div>
//           </Col>



//           {/* IMAGE PANEL */}
//           <Col xs={24} lg={12}>
//             <div style={{ ...cardStyle, display: "flex", flexDirection: "column", overflow: "hidden" }}>
//               <div style={{ flex: 1, display: "flex", justifyContent: "center", alignItems: "center", overflow: "hidden", color: darkMode ? "#c7c7c7" : "#666" }}>
//                 {imageUrl ? (
//                   <img src={imageUrl} alt="uploaded" style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }} />
//                 ) : (
//                   <p>No Image Uploaded</p>
//                 )}
//               </div>

//               <div style={{ padding: "10px", display: "flex", gap: "10px", borderTop: darkMode ? "1px solid #2b2e55" : "1px solid #ddd", background: darkMode ? "#1a1f4a" : "#f9f9f9" }}>
//                 <Row style={{width:"100%"}}>
//                   <Col span={8}>
//                     <Upload accept="image/*" showUploadList={false} customRequest={handleUpload} style={{width:"100%"}}>
//                       <Button icon={<UploadOutlined />} style={{ width: "90%" }}>Upload Image</Button>
//                     </Upload>
//                   </Col>
//                   <Col span={8}>
//                     <Button danger onClick={() => setImageUrl(null)} style={{ width: "90%" }}>Remove Image</Button>
//                   </Col>
//                   <Col span={8}>
//                     <InputNumber placeholder="Spatial resolution" step={0.1} style={{ width: "90%" }} onChange={setResolution} />
//                   </Col>
//                 </Row>
  
                
  
//               </div>
//             </div>
//           </Col>
          
//         </Row>
//       </div>
//     </div>
//   );
// };

// export default Home;
import { ThemeContext } from "./utils/style";
import { useContext, useState, useEffect, useRef } from "react";
import { Row, Col, Upload, Button, Input, Select, message as er, InputNumber, Switch, Modal } from "antd";
import { UploadOutlined, SendOutlined, PlusOutlined, MenuOutlined } from "@ant-design/icons";
import axios from "axios";
import { v4 as uuidv4 } from "uuid";

const { Option } = Select;

const Home = () => {
  const { darkMode, toggleDarkMode } = useContext(ThemeContext);

  const [imageUrl, setImageUrl] = useState(null);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState(() => {
    const stored = sessionStorage.getItem("storedMessages");
    return stored ? JSON.parse(stored) : [];
  });
  const [queryType, setQueryType] = useState("Select query type");
  const [resolution, setResolution] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [newImg, setNewImg] = useState(false);
  const [sidebar, setSidebar] = useState(false);
  const messagesEndRef = useRef(null);
  const [isModalVisible, setIsModalVisible] = useState(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const showModal = () => setIsModalVisible(true);
  const handleCancel = () => setIsModalVisible(false);


  const generateSessionId = () => {
    if(sessionStorage.getItem("currentSessionId")){
      return;
    }
    const sessionId = uuidv4();
    sessionStorage.setItem("currentSessionId", sessionId);
  }

  const generateNewChat = () => {
    setImageUrl(null);
    setInput("");
    setMessages([]);
    setQueryType("Select query type");
    setResolution(null);
    sessionStorage.removeItem("currentSessionId");
    sessionStorage.removeItem("uploadedImage");
  }

  const generateSessionByImageUpload = () => {
    setInput("");
    sessionStorage.removeItem("currentSessionId");
    generateSessionId();
  }

  const handleUpload = ({ file, onSuccess }) => {
    const reader = new FileReader();
    reader.onload = () => {
      setImageUrl(reader.result);
      //setMessages([]);
      setNewImg(true);
      sessionStorage.setItem("uploadedImage",reader.result);
      generateSessionByImageUpload();
      onSuccess("ok");
    };
    reader.readAsDataURL(file);
  };

  const handleSend = async () => {
    const token = sessionStorage.getItem("token");
    if(!token){
      er.error("Authentication failed.")
      return;
    }
    if(!sessionStorage.getItem("currentSessionId")){
      er.error("No session id found.");
      return;
    }
    if (!queryType || queryType==="Select query type") {
      er.error("Please select a query type.");
      return;
    }
    if (!imageUrl) {
      er.error("Please upload an image.");
      return;
    }
    if (!resolution) {
      er.error("Please enter the resolution.");
      return;
    }

    try {
      setMessages(prev => [...prev, { role: "user",type:"text", content: input }]);
      setInput("");

      const formData = new FormData();
      formData.append("session_id", sessionStorage.getItem("currentSessionId"));
      formData.append("query", input);
      formData.append("query_type", queryType);
      formData.append("spatial_resolution_m", parseFloat(resolution));
      if(newImg){
        const blob = await (await fetch(imageUrl)).blob();
        formData.append("image", blob, "uploaded_image.png");
      }
      else{
        formData.append("image_url",imageUrl);
      }

      const res = await axios.post("http://localhost:5000/api/chat", formData, {
        headers: { "Content-Type": "multipart/form-data", "Authorization" : `Bearer ${token}` },
      });

      setMessages(prev => [...prev, { role: "assistant",type:"text",content: res.data.reply }]);
      console.log(res.data.image);
      if(res.data.image){
        setMessages(prev=> [...prev, {role: "assistant", type:"image", content: res.data.image}]);
      }
      setNewImg(false);
      console.log(res.data.response);
    } catch (e) {
      console.log(e.message);
      er.error(e.message);
    }
  };

  const getChatHistory = async () => {
    try{
      const token = sessionStorage.getItem("token");
      console.log(token);
      const res = await axios.get("http://localhost:5000/api/chat-history",{
        headers:{"Authorization":`Bearer ${token}`}
      });
      console.log(res.data);
      setSessions(res.data);
    }
    catch(e){
      console.log(e.message);
      er.error(e.message);
    }
  }

  const handleHistory = async (sessionId) => {
    console.log(sessionId);
    try{
      const token = sessionStorage.getItem("token");
      sessionStorage.removeItem("currentSessionId");
      sessionStorage.setItem("currentSessionId", sessionId);
      const res = await axios.get(`http://localhost:5000/api/get-history`, {
        headers: { "Authorization": `Bearer ${token}` },
        params: { sessionId: sessionId }
      });
      setImageUrl(res.data.image.image_url);
      setMessages(res.data.messages)
      console.log(res.data);
    }
    catch(e){
      er.error(e.message);
    }
  }


  useEffect(() => {
    sessionStorage.setItem("storedMessages", JSON.stringify(messages));
  }, [messages]);

  useEffect(() => {
    if (imageUrl) sessionStorage.setItem("uploadedImage", imageUrl);
  }, [imageUrl]);

  useEffect(() => {
    const storedImage = sessionStorage.getItem("uploadedImage");
    if (storedImage) setImageUrl(storedImage);
    getChatHistory();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const chatContainerRef = useRef(null);


  useEffect(() => {
  if (chatContainerRef.current) {
    chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    // OR scroll by fixed pixels:
    // chatContainerRef.current.scrollTop += 200; // scroll down 200px per message
  }
}, [messages]);



  const [height, setHeight] = useState(window.innerWidth >= 992 ? "85vh" : "35vh");
  useEffect(() => {
    const handleResize = () => setHeight(window.innerWidth >= 992 ? "85vh" : "35vh");
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const cardStyle = {
    background: darkMode ? "#1a1b2a" : "#ffffff",
    height,
    border: darkMode ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0, 0, 0, 0.1)",
    boxShadow: darkMode
      ? "0 8px 20px rgba(0,0,0,0.5)"
      : "0 8px 20px rgba(0,0,0,0.15)",
    transition: "0.3s ease",
    backdropFilter: "blur(6px)",
  };

  const outerCard = {
    background: darkMode ? "#1a1b2a" : "#ffffff",
    border: darkMode ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.08)",
    boxShadow: darkMode
      ? "0 8px 20px rgba(0,0,0,0.5)"
      : "0 8px 20px rgba(0,0,0,0.15)",
    transition: "0.3s ease",
    backdropFilter: "blur(6px)",
  };

  return (
    <>
        <div style={{ height:"100vh",backgroundColor: darkMode ? "#0f111a" : "#f7f7f8", overflow:"hidden"}}>
            <Row style={{width:"100vw"}}>
                <Col xs={24} lg={sidebar?4:1} style={{ height:"100vh"}}>
                    {sidebar?
                        <div style={{ ...cardStyle, display: "flex", flexDirection: "column", position: "relative", bottom:"0", left:"0", height:"100vh" }}>
                        <div style={{display:"flex", justifyContent:"space-between", alignItems:"center", padding:"0.5rem 1rem 0.5rem 1rem", color: darkMode ? "#c7c7c7" : "#111827",}}>
                            <h4>New chat</h4>
                            <div>
                            <Button 
                                onClick={generateNewChat}
                                style={{
                                height: "36px",
                                padding: "0 15px",
                                borderRadius: "8px",
                                background: "#4549b8",
                                border: "none",
                                fontWeight: 600,
                                color:"#fff"
                                }}
                                icon={<PlusOutlined />}
                            ></Button>
                            <Button
                                onClick={() => setSidebar(!sidebar)}

                                style={{
                                    height: "36px",
                                    padding: "0px 5px",
                                    margin:"0.5rem 0 0 1rem",
                                    borderRadius: "8px",
                                    //background: darkMode ? "#1a1b2a" : "#ffffff",
                                    background: "#4549b8",
                                    //color: darkMode ? "#c7c7c7" : "#111827",
                                    border: "none",
                                    fontWeight: 600,
                                    color:"#fff"
                                }}
                                icon={<MenuOutlined />}
                            ></Button>
                        </div>
                        </div>
                        <div style={{ padding: "1rem", color: darkMode ? "#c7c7c7" : "#111827", display:"flex", flexDirection:"row", justifyContent:"space-between", alignItems:"center" }}>
                            <h4>Chat history</h4>
                        </div>
                        <div style={{ flex: 1, overflowY: "auto", padding: "0.5rem", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                            {sessions.map((session) => (
                            <div key={session.sessionId} style={{ display: "flex", justifyContent:"start"}}>
                                <div style={{
                                    width: "100%",
                                    padding: "10px 14px",
                                    borderRadius: "15px",
                                    background: darkMode ? "#2f335c" : "#e5e5e5",
                                    color: darkMode ? "#fff" : "#000",
                                    wordWrap: "break-word",
                                    whiteSpace: "pre-wrap",
                                    cursor:"pointer",
                                }}
                                onClick={() => handleHistory(session.sessionId)}
                                onMouseEnter={(e) => (e.currentTarget.style.opacity = 0.8)}
                                onMouseLeave={(e) => (e.currentTarget.style.opacity = 1)}
                                >
                                {session.title}
                                </div>
                            </div>
                            ))}
                        </div>
                        </div>
                        :<div style={{ ...cardStyle, display: "flex", flexDirection: "column", position: "relative", alignItems:"center",bottom:"0", left:"0" , height:"100vh"}}>
                        <Button
                            onClick={() => setSidebar(!sidebar)}
                            style={{
                                height: "36px",
                                padding: "0px 5px",
                                margin:"0.5rem 0 0 0",
                                borderRadius: "8px",
                                //background: darkMode ? "#1a1b2a" : "#ffffff",
                                background: "#4549b8",
                                //color: darkMode ? "#c7c7c7" : "#111827",
                                border: "none",
                                fontWeight: 600,
                                color:"#fff"
                            }}
                            icon={<MenuOutlined />}
                        ></Button>
                        </div>
                    }
                </Col>
                <Col xs={24} lg={sidebar?20:23}>
                    <Row style={{width:"100%"}}>
                        <div
                            style={{
                            backgroundColor: darkMode ? "#0f111a" : "#f7f7f8",
                            padding: "0.5rem 1rem 0.5rem 1rem",
                            width:"100%",
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                            borderBottom: darkMode ? "1px solid #1f2937" : "1px solid #e5e7eb",
                            boxShadow: darkMode
                                ? "0 2px 8px rgba(0,0,0,0.4)"
                                : "0 2px 8px rgba(0,0,0,0.1)",
                            }}
                        >
                            <div style={{ display: "flex", alignItems: "center", gap: "1rem", color: darkMode ? "#fff" : "#111827" }}>
                            <div>
                                <span style={{ margin: 0, fontSize:"1.3rem",display:"inline", fontWeight:"bold" }}>geoNLI</span>
                                <span style={{ margin: 0, fontSize:"1.3rem" }}> Image Chat</span>
                            </div>
                            </div>
                            <Switch
                            style={{backgroundColor:"#4549b8"}}
                            checked={darkMode}
                            onChange={toggleDarkMode}
                            />

                        </div>
                    </Row>
                    <Row style={{width:"100%", height:"100vh",display:"flex", justifyContent:"center"}}>
                        <Col span={24} style={{height:"100%",display:"flex", justifyContent:"center"}}>
                          <div
                            style={{
                              ...outerCard,
                              // position:"relative",
                              // bottom:"1rem",
                              width:"100%",
                              //margin:"1rem",
                              padding: "15px",
                              //backgroundColor: darkMode ? "#0f111a" : "#f7f7f8",
                              background: darkMode ? "#1a1b2a" : "#ffffff",
                            }}
                          >
                            <Row style={{width:"100%"}}>
                              {/* <Col xs={24} lg={10} style={{marginRight:"1rem"}}>
                                <div style={{ ...cardStyle, height: "85vh", display: "flex", flexDirection: "column", position: "relative" , boxShadow:"none"}}>

                                
                                  <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                                    {messages.map((msg, idx) => (
                                      <div key={idx} style={{ display: "flex", justifyContent: msg.role === "user" ? "flex-end" : "flex-start" }}>
                                        <div style={{
                                          maxWidth: "70%",
                                          padding: "10px 14px",
                                          borderRadius: "15px",
                                          background: msg.role === "user" ? "#4549b8" : darkMode ? "#2f335c" : "#e5e5e5",
                                          color: msg.role === "user" ? "#fff" : darkMode ? "#fff" : "#000",
                                          wordWrap: "break-word",
                                          whiteSpace: "pre-wrap"
                                        }}>
                                          {msg.content}
                                        </div>
                                      </div>
                                    ))}
                                  </div>

                                  <div style={{
                                      display: "flex",
                                      padding: "10px",
                                      border: darkMode ? "1px solid #2b2e55" : "1px solid #ddd",
                                      gap: "10px",
                                      position: "absolute",
                                      bottom: "0rem",
                                      width: "100%",
                                      boxSizing: "border-box",
                                      background: darkMode ? "#1a1f4a" : "#f9f9f9",
                                      // borderBottomLeftRadius:"15px",
                                      // borderBottomRightRadius:"15px",
                                      borderRadius:"15px",
                                      height:"auto"
                                    }}
                                  >
                                      <Input
                                        placeholder="Ask something..."
                                        value={input}
                                        onChange={(e) => setInput(e.target.value)}
                                        style={{
                                          flex: 1,
                                          height: "36px",
                                          borderRadius: "15px",
                                          border: darkMode ? "1px solid #2b2e55" : "1px solid #ccc",
                                          background: darkMode ? "#0f112e" : "#fff",
                                          color: darkMode ? "#fff" : "#000",
                                        }}
                                      />
                                      <Select
                                        placeholder="Select query type"
                                        value={queryType}
                                        onChange={setQueryType}
                                        style={{
                                          width: "30%",
                                          height:"36px",
                                          margin: "0 0",
                                          background: darkMode ? "#0f112e" : "#fafafa",
                                          color: darkMode ? "#c7c7c7" : "#111827",
                                          border: darkMode ? "1px solid #2b2e55" : "1px solid #ddd",
                                          borderRadius:"15px"
                                        }}
                                      >
                                        <Option value="caption_query">Caption query</Option>
                                        <Option value="grounding_query">Grounding query</Option>
                                        <Option value="attribute_binary_query">Attribute binary query</Option>
                                        <Option value="attribute_numeric_query">Attribute numeric query</Option>
                                        <Option value="attribute_semantic_query">Attribute semantic query</Option>
                                      </Select>
                                      <Button
                                        onClick={handleSend}
                                        style={{
                                          height: "36px",
                                          padding: "0 25px",
                                          borderRadius: "15px",
                                          background: "#4549b8",
                                          border: "none",
                                          fontWeight: 600,
                                          color:"#fff"
                                        }}
                                        icon={<SendOutlined />}
                                      >
                                      </Button>
                                    </div>


                                </div>
                              </Col> */}
                              <Col xs={24} lg={10} style={{ marginRight: "1rem" }}>
  <div
    style={{
      ...cardStyle,
      height: "85vh",
      display: "flex",
      flexDirection: "column",
      position: "relative",
      boxShadow: "none",
      padding: "1rem", // space between border and content
      borderRadius: "15px",
      //background: darkMode ? "#1a1f4a" : "#fff",
    }}
  >
    {/* Messages container */}
    <div
      className="message-container"
      ref={chatContainerRef}
      style={{
        flex: 1,
        //height:"70vh",
        overflowY: "auto",
        display: "flex",
        flexDirection: "column",
        gap: "0.5rem",
        marginBottom: "1rem",
      }}
    >
      {messages.map((msg, idx) => (
        <div
          key={idx}
          style={{
            display: "flex",
            justifyContent: msg.role === "user" ? "flex-end" : "flex-start",
          }}
        >
          {msg.type=="text" && <div
            style={{
              maxWidth: "70%",
                                          padding: "10px 14px",
                                          borderRadius: "15px",
                                          background: msg.role === "user" ? "#4549b8" : darkMode ? "#2f335c" : "#e5e5e5",
                                          color: msg.role === "user" ? "#fff" : darkMode ? "#fff" : "#000",
                                          wordWrap: "break-word",
                                          whiteSpace: "pre-wrap"
            }}
          >
            {msg.content}
          </div>}
          {msg.type=="image" && <img
            src={msg.content}
            alt="response"
            style={{ width: "70%", borderRadius: "10px", marginTop: "6px" }}
          />}
        </div>
      ))}
      {/* <div ref={messagesEndRef} /> */}
    </div>

    {/* Input + Select + Button */}
    <div
      style={{
        display: "flex",
        padding: "10px",
        border: darkMode ? "1px solid #2b2e55" : "1px solid #ddd",
        gap: "10px",
        borderRadius: "15px",
        background: darkMode ? "#1a1f4a" : "#f9f9f9",
      }}
    >
      <Input
        placeholder="Ask something..."
        value={input}
        onChange={(e) => setInput(e.target.value)}
        style={{
          flex: 1,
          height: "36px",
          borderRadius: "15px",
          border: darkMode ? "1px solid #2b2e55" : "1px solid #ccc",
          background: darkMode ? "#0f112e" : "#fff",
          color: darkMode ? "#fff" : "#000",
        }}
      />
      <Select
        placeholder="Select query type"
        value={queryType}
        onChange={setQueryType}
        style={{
          width: "30%",
          height: "36px",
          background: darkMode ? "#0f112e" : "#fafafa",
          color: darkMode ? "#c7c7c7" : "#111827",
          border: darkMode ? "1px solid #2b2e55" : "1px solid #ddd",
          borderRadius: "15px",
        }}
      >
        <Option value="caption_query">Caption query</Option>
        <Option value="grounding_query">Grounding query</Option>
        <Option value="attribute_binary_query">Attribute binary query</Option>
        <Option value="attribute_numeric_query">Attribute numeric query</Option>
        <Option value="attribute_semantic_query">Attribute semantic query</Option>
      </Select>
      <Button
        onClick={handleSend}
        style={{
          height: "36px",
          padding: "0 25px",
          borderRadius: "15px",
          background: "#4549b8",
          border: "none",
          fontWeight: 600,
          color: "#fff",
        }}
        icon={<SendOutlined />}
      />
    </div>
  </div>
</Col>




                              {/* IMAGE PANEL */}
                              <Col xs={24} lg={13}>
                                <div style={{ ...cardStyle, display: "flex", flexDirection: "column", overflow: "hidden", boxShadow:"none", padding:"1rem", borderRadius:"15px" }}>
                                  <div style={{ flex: 1, display: "flex", justifyContent: "center", alignItems: "center", overflow: "hidden", color: darkMode ? "#c7c7c7" : "#666" }}>
                                    {imageUrl ? (
                                      <img src={imageUrl} alt="uploaded" style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }} />
                                    ) : (
                                      <p>No Image Uploaded</p>
                                    )}
                                  </div>

                                  <div style={{ padding: "10px", display: "flex", gap: "10px", border: darkMode ? "1px solid #2b2e55" : "1px solid #ddd", background: darkMode ? "#1a1f4a" : "#f9f9f9",borderRadius:"15px" }}>
                                    <Row style={{width:"100%", borderRadius:"15px"}}>
                                      <Col span={8}>
                                        <Upload accept="image/*" showUploadList={false} customRequest={handleUpload} style={{width:"100%"}}>
                                          <Button icon={<UploadOutlined />} style={{ width: "90%", borderRadius:"15px", height:"36px" ,background: darkMode ? "#0f112e" : "#fff",color: darkMode ? "#c7c7c7" : "#000",border: darkMode ? "1px solid #2b2e55" : "1px solid #ddd",}}>Upload Image</Button>
                                        </Upload>
                                        <Modal
                                          title="Upload Image"
                                          visible={isModalVisible}
                                          onCancel={handleCancel}
                                          footer={[
                                            <Button key="cancel" onClick={handleCancel}>
                                              Cancel
                                            </Button>,
                                            <Button key="attach" type="primary" onClick={()=>{handleUpload;setIsModalVisible(false)}}>
                                              Proceed
                                            </Button>,
                                          ]}
                                        >
                                          <p>This will start a new chat. Do you want to proceed?</p>
                                        </Modal>
                                      </Col>
                                      <Col span={8}>
                                        <Button danger onClick={() => setImageUrl(null)} style={{ width: "90%",borderRadius:"15px",height:"36px",background: darkMode ? "#0f112e" : "#fff",color: darkMode ? "#c7c7c7" : "#000",border: darkMode ? "1px solid #2b2e55" : "1px solid #ddd", }}>Remove Image</Button>
                                      </Col>
                                      <Col span={8}>
                                        <InputNumber placeholder="Spatial resolution" className={darkMode?"res-dark":"res-light"} step={0.1} style={{ width: "90%",borderRadius:"15px",height:"36px",background: darkMode ? "#0f112e" : "#fff",color: darkMode ? "#fff" : "#000",border: darkMode ? "1px solid #2b2e55" : "1px solid #ddd", }} onChange={setResolution} />
                                      </Col>
                                    </Row>
                      
                                    
                      
                                  </div>
                                </div>
                              </Col>
                            </Row>
                          </div>
                        </Col>
                        
                    </Row>
                </Col>
            </Row>
        </div>
    </>
  );
};

export default Home;
