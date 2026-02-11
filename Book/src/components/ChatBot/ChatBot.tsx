import React, { useState, useRef, useEffect, useCallback } from 'react';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import TextSelectionPopup from './TextSelectionPopup';
import styles from './chatbot.module.css';

const API_BASE = typeof window !== 'undefined'
  ? (window as any).__CHATBOT_API_URL || 'http://localhost:8000'
  : 'http://localhost:8000';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Array<{ chapter: string; section: string; score: number }>;
}

export default function ChatBot() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = useCallback(async (message: string) => {
    // Add user message
    setMessages(prev => [...prev, { role: 'user', content: message }]);
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, session_id: sessionId }),
      });

      if (!response.ok) throw new Error('Chat request failed');

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');

      const decoder = new TextDecoder();
      let assistantContent = '';
      let sources: Message['sources'] = [];

      // Add empty assistant message for streaming
      setMessages(prev => [...prev, { role: 'assistant', content: '', sources: [] }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.type === 'token') {
                assistantContent += data.content;
                setMessages(prev => {
                  const updated = [...prev];
                  updated[updated.length - 1] = {
                    role: 'assistant',
                    content: assistantContent,
                    sources,
                  };
                  return updated;
                });
              } else if (data.type === 'sources') {
                sources = data.sources;
              } else if (data.type === 'session') {
                setSessionId(data.session_id);
              }
            } catch {
              // Skip malformed JSON
            }
          }
        }
      }

      // Final update with sources
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: 'assistant',
          content: assistantContent,
          sources,
        };
        return updated;
      });
    } catch (error) {
      setMessages(prev => [
        ...prev.filter(m => m.content !== ''),
        {
          role: 'assistant',
          content: 'Sorry, I encountered an error connecting to the server. Please make sure the backend is running.',
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }, [sessionId]);

  const handleSelectionAsk = useCallback((selectedText: string) => {
    setIsOpen(true);
    // Add context message and prompt
    const contextMessage = `I selected this text from the book:\n\n"${selectedText.slice(0, 300)}${selectedText.length > 300 ? '...' : ''}"\n\nPlease explain this passage.`;

    // Use selection endpoint
    (async () => {
      setMessages(prev => [...prev, { role: 'user', content: `Explain selected text: "${selectedText.slice(0, 100)}..."` }]);
      setIsLoading(true);

      try {
        const response = await fetch(`${API_BASE}/api/chat/selection`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            selected_text: selectedText,
            question: 'Please explain this passage in simple terms.',
            session_id: sessionId,
          }),
        });

        if (!response.ok) throw new Error('Selection request failed');

        const reader = response.body?.getReader();
        if (!reader) throw new Error('No response body');

        const decoder = new TextDecoder();
        let assistantContent = '';
        let sources: Message['sources'] = [];

        setMessages(prev => [...prev, { role: 'assistant', content: '', sources: [] }]);

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.type === 'token') {
                  assistantContent += data.content;
                  setMessages(prev => {
                    const updated = [...prev];
                    updated[updated.length - 1] = {
                      role: 'assistant',
                      content: assistantContent,
                      sources,
                    };
                    return updated;
                  });
                } else if (data.type === 'sources') {
                  sources = data.sources;
                } else if (data.type === 'session') {
                  setSessionId(data.session_id);
                }
              } catch {
                // Skip
              }
            }
          }
        }

        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: 'assistant',
            content: assistantContent,
            sources,
          };
          return updated;
        });
      } catch {
        setMessages(prev => [
          ...prev.filter(m => m.content !== ''),
          {
            role: 'assistant',
            content: 'Sorry, I encountered an error. Please make sure the backend is running.',
          },
        ]);
      } finally {
        setIsLoading(false);
      }
    })();
  }, [sessionId]);

  return (
    <>
      <TextSelectionPopup onAskAboutSelection={handleSelectionAsk} />

      {/* Floating chat button */}
      <button
        className={styles.chatButton}
        onClick={() => setIsOpen(!isOpen)}
        title={isOpen ? 'Close chat' : 'Ask AI about this book'}
      >
        {isOpen ? '\u2715' : '\uD83E\uDD16'}
      </button>

      {/* Chat panel */}
      {isOpen && (
        <div className={styles.chatPanel}>
          <div className={styles.chatHeader}>
            <span className={styles.chatTitle}>Physical AI Book Assistant</span>
            <button className={styles.closeButton} onClick={() => setIsOpen(false)}>
              &#10005;
            </button>
          </div>

          <div className={styles.messagesContainer}>
            {messages.length === 0 && (
              <div className={styles.welcomeMessage}>
                Welcome! I'm your AI assistant for the Physical AI book.
                Ask me anything about robotics, ROS 2, simulation, or any topic covered in the chapters.
                <br /><br />
                You can also <strong>select text</strong> in the book and click "Ask AI about this" for explanations.
              </div>
            )}
            {messages.map((msg, i) => (
              <ChatMessage
                key={i}
                role={msg.role}
                content={msg.content}
                sources={msg.sources}
              />
            ))}
            {isLoading && messages[messages.length - 1]?.content === '' && (
              <div className={styles.typing}>
                <div className={styles.typingDot} />
                <div className={styles.typingDot} />
                <div className={styles.typingDot} />
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <ChatInput onSend={handleSend} disabled={isLoading} />
        </div>
      )}
    </>
  );
}
