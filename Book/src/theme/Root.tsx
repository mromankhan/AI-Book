import React from 'react';
import AuthProvider from '@site/src/components/Auth/AuthProvider';
import ChatBot from '@site/src/components/ChatBot/ChatBot';

export default function Root({ children }: { children: React.ReactNode }) {
  return (
    <AuthProvider>
      {children}
      <ChatBot />
    </AuthProvider>
  );
}
