import React, { useState, useRef, useLayoutEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [history, setHistory] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const chatEndRef = useRef(null);

  useLayoutEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [history, isTyping]);

  const getTimestamp = () => {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const trimmedQuery = query.trim();
    if (!trimmedQuery) return;

    const userMessage = { sender: 'user', text: trimmedQuery, time: getTimestamp() };
    setHistory((prev) => [...prev, userMessage]);
    setQuery('');
    setIsTyping(true);

    try {
      const res = await axios.post('http://localhost:8000/chat', { message: trimmedQuery });
      const botMessage = { sender: 'bot', text: res.data.response, time: getTimestamp() };
      setHistory((prev) => [...prev, botMessage]);
    } catch (err) {
      console.error('API Error:', err);
      setHistory((prev) => [
        ...prev,
        {
          sender: 'bot',
          text: '⚠️ Sorry, something went wrong. Please try again later.',
          time: getTimestamp()
        },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="App">
      <h1>🤖 Customer Service Assistant</h1>
      <div className="chat-box">
        {history.map((msg, i) => (
          <div key={i} className={`message-row ${msg.sender}`}>
            <div className="message">
              <div className="text">{msg.text}</div>
              <div className="timestamp">{msg.time}</div>
            </div>
          </div>
        ))}
        {isTyping && (
          <div className="message-row bot">
            <div className="message typing">
              <span className="dot"></span>
              <span className="dot"></span>
              <span className="dot"></span>
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      <form onSubmit={handleSubmit}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Type your message and hit Enter..."
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              handleSubmit(e);
            }
          }}
        />
        <button type="submit">Send</button>
      </form>
    </div>
  );
}

export default App;
