import React from 'react';
import styles from './chatbot.module.css';

interface Source {
  chapter: string;
  section: string;
  score: number;
}

interface ChatMessageProps {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
}

export default function ChatMessage({ role, content, sources }: ChatMessageProps) {
  const isUser = role === 'user';

  return (
    <div className={`${styles.message} ${isUser ? styles.userMessage : styles.assistantMessage}`}>
      <div dangerouslySetInnerHTML={{ __html: simpleMarkdown(content) }} />
      {sources && sources.length > 0 && (
        <div className={styles.sources}>
          <strong>Sources:</strong>
          {sources.map((src, i) => (
            <div key={i} className={styles.sourceItem}>
              {src.chapter} - {src.section}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function simpleMarkdown(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/\n/g, '<br/>');
}
