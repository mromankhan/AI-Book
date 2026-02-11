import React, { useState } from 'react';
import PersonalizeButton from './PersonalizeButton';
import TranslateButton from './TranslateButton';
import styles from './chapterActions.module.css';

interface ChapterToolbarProps {
  chapter: string;
}

export default function ChapterToolbar({ chapter }: ChapterToolbarProps) {
  const [personalizedContent, setPersonalizedContent] = useState<string | null>(null);
  const [translatedContent, setTranslatedContent] = useState<string | null>(null);

  return (
    <div>
      <div className={styles.toolbar}>
        <span className={styles.toolbarLabel}>Chapter tools:</span>
        <PersonalizeButton
          chapter={chapter}
          onPersonalized={setPersonalizedContent}
          isActive={!!personalizedContent}
          onReset={() => setPersonalizedContent(null)}
        />
        <TranslateButton
          chapter={chapter}
          onTranslated={setTranslatedContent}
          isActive={!!translatedContent}
          onReset={() => setTranslatedContent(null)}
        />
      </div>

      {personalizedContent && (
        <div className={styles.personalizedContent}>
          <span className={styles.personalizedBadge}>Personalized for you</span>
          <div dangerouslySetInnerHTML={{ __html: simpleMarkdown(personalizedContent) }} />
        </div>
      )}

      {translatedContent && (
        <div className={styles.translatedContent}>
          <span className={styles.translatedBadge}>اردو ترجمہ</span>
          <div dangerouslySetInnerHTML={{ __html: simpleMarkdown(translatedContent) }} />
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
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    .replace(/\n/g, '<br/>');
}
