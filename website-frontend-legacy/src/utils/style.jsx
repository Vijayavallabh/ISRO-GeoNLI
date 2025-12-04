import React, { createContext, useState } from "react";

export const ThemeContext = createContext();

export const ThemeProvider = ({ children }) => {
	const [darkMode, setDarkMode] = useState(true);
	const toggleDarkMode = () => {
		setDarkMode((prevMode) => !prevMode);
	};

	return (
		<ThemeContext.Provider
			value={{ darkMode, toggleDarkMode }}
		>
			{children}
		</ThemeContext.Provider>
	);
};