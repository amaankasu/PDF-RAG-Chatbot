import React, { useState, useRef, useEffect } from 'react';
import { GoLaw } from 'react-icons/go';
import './App.css';
import ReactMarkdown from 'react-markdown';

// Improved ReasoningDropdown component (ChatGPT style)
const ReasoningDropdown = ({ reasoningTime, finalContext }) => {
  const [isExpanded, setIsExpanded] = useState(true);  // Default to expanded

  return (
    <div className="reasoning-container">
      <div 
        className="reasoning-header"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <span className="reasoning-label">Reasoned for {reasoningTime} seconds</span>
        <span className="dropdown-icon-container">
          {isExpanded ? '—' : '+'}
        </span>
      </div>
      {isExpanded && (
        <div className="reasoning-content">
          <ReactMarkdown components={{
            p: ({node, ...props}) => <p className="markdown-paragraph" {...props} />,
            ul: ({node, ...props}) => <ul className="markdown-list" {...props} />,
            ol: ({node, ...props}) => <ol className="markdown-list" {...props} />,
            li: ({node, ...props}) => <li className="markdown-list-item" {...props} />
          }}>
            {finalContext}
          </ReactMarkdown>
        </div>
      )}
    </div>
  );
};

function ChatbotUI() {
  // Initialize with empty data for dynamic updates.
  const [messages, setMessages] = useState([]);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [uploadedPdf, setUploadedPdf] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  
  const fileInputRef = useRef(null);
  const dropAreaRef = useRef(null);
  const chatContentRef = useRef(null); // Add reference to chat content for scrolling

  // Fixed backend URL for advanced model
  const backendUrl = 'http://localhost:5000';

  // Add a state to track which message is being copied
  const [copiedMessageId, setCopiedMessageId] = useState(null);

  // Setup drag and drop event listeners
  useEffect(() => {
    const chatContent = dropAreaRef.current;
    
    const handleDragEnter = (e) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(true);
    };
    
    const handleDragOver = (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (!isDragging) setIsDragging(true);
    };
    
    const handleDragLeave = (e) => {
      e.preventDefault();
      e.stopPropagation();
      
      // Check if mouse is outside the drop area
      const rect = chatContent.getBoundingClientRect();
      const x = e.clientX;
      const y = e.clientY;
      
      if (
        x < rect.left ||
        x >= rect.right ||
        y < rect.top ||
        y >= rect.bottom
      ) {
        setIsDragging(false);
      }
    };
    
    const handleDrop = (e) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
      
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        processFile(files[0]);
      }
    };
    
    if (chatContent) {
      chatContent.addEventListener('dragenter', handleDragEnter);
      chatContent.addEventListener('dragover', handleDragOver);
      chatContent.addEventListener('dragleave', handleDragLeave);
      chatContent.addEventListener('drop', handleDrop);
      
      return () => {
        chatContent.removeEventListener('dragenter', handleDragEnter);
        chatContent.removeEventListener('dragover', handleDragOver);
        chatContent.removeEventListener('dragleave', handleDragLeave);
        chatContent.removeEventListener('drop', handleDrop);
      };
    }
  }, [isDragging]);

  const processFile = (file) => {
    if (file && file.type === 'application/pdf') {
      const displayName = file.name.length > 30 ? file.name.substring(0, 27) + '...' : file.name;
      setUploadedPdf({
        name: displayName,
        type: 'PDF'
      });
      
      console.log(`Attempting to upload file to ${backendUrl}/upload`);
      
      // Create a FormData to send the file
      const formData = new FormData();
      formData.append('file', file);
      
      // Upload the file to the backend
      fetch(`${backendUrl}/upload`, {
        method: 'POST',
        body: formData,
      })
      .then(response => {
        console.log('Response received:', response.status, response.statusText);
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status} ${response.statusText}`);
        }
        return response.json();
      })
      .then(data => {
        console.log('Upload successful:', data);
        // Ensure the UI state is updated correctly
        setUploadedPdf({
          name: displayName,
          type: 'PDF'
        });
      })
      .catch(err => {
        console.error("Error uploading PDF:", err.message || err);
        // Show detailed error message
        const errorMessage = err.message || 'Unknown error occurred';
        console.error('Detailed error:', {
          url: `${backendUrl}/upload`,
          error: errorMessage
        });
        // Reset the UI if upload failed
        setUploadedPdf(null);
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
        alert(`Failed to upload PDF: ${errorMessage}`);
      });
    } else {
      alert("Please upload a PDF file");
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (file) {
      processFile(file);
    }
  };

  const handleDeletePdf = async () => {
    try {
      const response = await fetch(`${backendUrl}/delete-pdf`, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        console.error("Error deleting PDF:", errorData.detail || response.statusText);
        alert("Error removing PDF. Please try again.");
        return;
      }
      
      const data = await response.json();
      console.log(data.message);
      
      // Reset UI state regardless of backend response
      setUploadedPdf(null);
      
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (err) {
      console.error("Error deleting PDF:", err);
      // Even if there's an error with the backend, still reset the UI
      setUploadedPdf(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleSendMessage = async () => {
    if (currentQuestion.trim() === '') return;
    
    const queryText = currentQuestion;
    
    // Add user message
    const newUserMessage = {
      id: messages.length + 1,
      type: 'user',
      content: queryText,
    };
    setMessages([...messages, newUserMessage]);
    
    // Add a placeholder bot message with default "Analyzing" values
    const placeholderBotMessage = {
      id: messages.length + 2,
      type: 'bot',
      reasoningTime: 'Analyzing',
      content: 'Analyzing',
      finalContext: '',
    };
    setMessages(prevMessages => [...prevMessages, placeholderBotMessage]);
    setCurrentQuestion('');

    try {
      const response = await fetch(`${backendUrl}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // Send the query with model set to advanced (even if backend ignores it)
        body: JSON.stringify({ query: queryText, model: "advanced" }),
      });
      const data = await response.json();
      if (!response.ok) {
        console.error(data.error);
        // Update placeholder with error
        setMessages(prevMessages =>
          prevMessages.map(msg =>
            msg.id === placeholderBotMessage.id 
              ? { ...msg, content: data.error, reasoningTime: 0 }
              : msg
          )
        );
      } else {
        // Update placeholder with final answer, reasoning time, and final context
        setMessages(prevMessages =>
          prevMessages.map(msg =>
            msg.id === placeholderBotMessage.id 
              ? { 
                  ...msg, 
                  content: data.answer, 
                  reasoningTime: data.reasoningTime,
                  finalContext: data.finalContext 
                }
              : msg
          )
        );
      }
    } catch (err) {
      console.error("Error querying:", err);
      setMessages(prevMessages =>
        prevMessages.map(msg =>
          msg.id === placeholderBotMessage.id 
            ? { ...msg, content: "Error querying", reasoningTime: 0 }
            : msg
        )
      );
    }
  };
  
  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  // Function to scroll to the bottom of the chat
  const scrollToBottom = () => {
    if (chatContentRef.current) {
      chatContentRef.current.scrollTop = chatContentRef.current.scrollHeight;
    }
  };

  // Scroll to bottom whenever messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleCopyMessage = async (content) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedMessageId(content);
      setTimeout(() => setCopiedMessageId(null), 2000); // Reset after 2 seconds
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  // Render a formatted bot message with proper styling
  const renderBotMessage = (message) => {
    // Don't remove "Final Answer:" from the content
    let cleanedContent = message.content;
    
    return (
      <div className="bot-message-container">
        {/* Reasoning section - placed above the answer */}
        {message.reasoningTime !== 'Analyzing' && message.finalContext && (
          <ReasoningDropdown 
            reasoningTime={message.reasoningTime}
            finalContext={message.finalContext}
          />
        )}
        
        {/* No extra Final Answer label - just show the content directly */}
        <div className="bot-message">
          <ReactMarkdown components={{
            p: ({node, ...props}) => <p className="markdown-paragraph" {...props} />,
            ul: ({node, ...props}) => <ul className="markdown-list" {...props} />,
            ol: ({node, ...props}) => <ol className="markdown-list" {...props} />,
            li: ({node, ...props}) => <li className="markdown-list-item" {...props} />,
            h1: ({node, ...props}) => <h1 className="markdown-heading" {...props} />,
            h2: ({node, ...props}) => <h2 className="markdown-heading" {...props} />,
            h3: ({node, ...props}) => <h3 className="markdown-heading" {...props} />,
            h4: ({node, ...props}) => <h4 className="markdown-heading" {...props} />
          }}>
            {cleanedContent}
          </ReactMarkdown>
        </div>
        
        {/* Copy button */}
        <div className="message-actions">
          <button 
            className="copy-icon-button"
            onClick={() => handleCopyMessage(message.content)}
            title={copiedMessageId === message.content ? "Copied!" : "Copy"}
          >
            <span className="copy-button-icon"></span>
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className="chatbot-container">
      {/* Header */}
      <header className="chatbot-header">
        <div className="header-left">
          <div className="logo-container">
            <GoLaw size={32} color="#000" style={{ cursor: 'pointer' }} />
          </div>
          <div className="chatgpt-title">
            Legal RAG - Your Law Firm's Private, Reliable AI Assistant.
          </div>
        </div>
        <div className="header-right">
          {/* Share button and user avatar removed */}
        </div>
      </header>

      {/* Chat Content */}
      <div 
        className={`chat-content ${isDragging ? 'dragging' : ''}`}
        ref={(el) => { 
          dropAreaRef.current = el; 
          chatContentRef.current = el; 
        }}
      >
        {isDragging && (
          <div className="drop-overlay">
            <div className="drop-message">
              <div className="drop-icon"></div>
              <p>Drop your PDF here</p>
            </div>
          </div>
        )}
        
        {uploadedPdf && (
          <div className="pdf-attachment">
            <div className="pdf-icon"></div>
            <div className="pdf-details">
              <div className="pdf-name">{uploadedPdf.name}</div>
              <div className="pdf-type">{uploadedPdf.type}</div>
            </div>
            <button onClick={handleDeletePdf} className="pdf-delete-btn" title="Remove PDF">×</button>
          </div>
        )}
        
        {messages.map((message) => (
          <div key={message.id} className={`message-container ${message.type === 'user' ? 'user-message-container' : ''}`}>
            {message.type === 'user' ? (
              <div className="user-message">{message.content}</div>
            ) : message.reasoningTime === 'Analyzing' ? (
              <div className="analyzing-container">
                <div className="analyzing-spinner"></div>
                <div className="analyzing-text">Analyzing...</div>
              </div>
            ) : (
              renderBotMessage(message)
            )}
          </div>
        ))}
      </div>

      {/* Input Area */}
      <div className="input-area">
        <div className="input-container">
          <input
            type="text"
            className="message-input"
            placeholder="Enter your Query"
            value={currentQuestion}
            onChange={(e) => setCurrentQuestion(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
          />
          
          <div className="input-buttons">
            <button className="attachment-button" onClick={triggerFileInput}>
              <span className="plus-icon"></span>
            </button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept="application/pdf"
              style={{ display: 'none' }}
            />
            <button className="send-button" onClick={handleSendMessage}>
              <span className="send-icon"></span>
            </button>
          </div>
        </div>
      </div>
      
      <button className="scroll-bottom-button" onClick={scrollToBottom}>
        <span className="arrow-down-icon"></span>
      </button>
    </div>
  );
}

export default ChatbotUI;
