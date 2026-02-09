import React, { useState, useEffect, useCallback } from 'react';
import styles from './chatbot.module.css';

interface TextSelectionPopupProps {
  onAskAboutSelection: (selectedText: string) => void;
}

export default function TextSelectionPopup({ onAskAboutSelection }: TextSelectionPopupProps) {
  const [position, setPosition] = useState<{ x: number; y: number } | null>(null);
  const [selectedText, setSelectedText] = useState('');

  const handleMouseUp = useCallback(() => {
    // Small delay to ensure selection is complete
    setTimeout(() => {
      const selection = window.getSelection();
      const text = selection?.toString().trim();

      if (text && text.length > 10 && text.length < 5000) {
        const range = selection!.getRangeAt(0);
        const rect = range.getBoundingClientRect();

        setSelectedText(text);
        setPosition({
          x: rect.left + rect.width / 2,
          y: rect.top - 10,
        });
      } else {
        setPosition(null);
        setSelectedText('');
      }
    }, 10);
  }, []);

  const handleClick = useCallback(() => {
    if (selectedText) {
      onAskAboutSelection(selectedText);
      setPosition(null);
      setSelectedText('');
      window.getSelection()?.removeAllRanges();
    }
  }, [selectedText, onAskAboutSelection]);

  useEffect(() => {
    document.addEventListener('mouseup', handleMouseUp);
    return () => document.removeEventListener('mouseup', handleMouseUp);
  }, [handleMouseUp]);

  // Hide popup on scroll
  useEffect(() => {
    const handleScroll = () => {
      setPosition(null);
      setSelectedText('');
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  if (!position) return null;

  return (
    <div
      className={styles.selectionPopup}
      style={{
        left: position.x,
        top: position.y,
        transform: 'translate(-50%, -100%)',
      }}
      onClick={handleClick}
    >
      Ask AI about this
    </div>
  );
}
