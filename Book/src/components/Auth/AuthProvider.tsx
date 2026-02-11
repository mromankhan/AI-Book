import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';

const API_BASE = typeof window !== 'undefined'
  ? (window as any).__CHATBOT_API_URL || 'http://localhost:8000'
  : 'http://localhost:8000';

interface User {
  id: string;
  email: string;
  name: string;
  profile?: {
    programming_level: string;
    hardware_experience: string;
    education_level: string;
    interests: string[];
  };
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  signUp: (data: SignUpData) => Promise<void>;
  signIn: (email: string, password: string) => Promise<void>;
  signOut: () => void;
}

interface SignUpData {
  email: string;
  password: string;
  name: string;
  programming_level: string;
  hardware_experience: string;
  education_level: string;
  interests: string[];
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  token: null,
  isLoading: true,
  signUp: async () => {},
  signIn: async () => {},
  signOut: () => {},
});

export function useAuth() {
  return useContext(AuthContext);
}

export default function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Check stored token on mount
  useEffect(() => {
    const stored = localStorage.getItem('auth_token');
    if (stored) {
      setToken(stored);
      fetchUser(stored).finally(() => setIsLoading(false));
    } else {
      setIsLoading(false);
    }
  }, []);

  const fetchUser = async (authToken: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/auth/me`, {
        headers: { Authorization: `Bearer ${authToken}` },
      });
      if (res.ok) {
        const userData = await res.json();
        setUser(userData);
      } else {
        localStorage.removeItem('auth_token');
        setToken(null);
      }
    } catch {
      // Server not available - keep token for later
    }
  };

  const signUp = useCallback(async (data: SignUpData) => {
    const res = await fetch(`${API_BASE}/api/auth/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Signup failed');
    }
    const { token: newToken, user: newUser } = await res.json();
    setToken(newToken);
    setUser(newUser);
    localStorage.setItem('auth_token', newToken);
  }, []);

  const signIn = useCallback(async (email: string, password: string) => {
    const res = await fetch(`${API_BASE}/api/auth/signin`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Sign in failed');
    }
    const { token: newToken, user: newUser } = await res.json();
    setToken(newToken);
    setUser(newUser);
    localStorage.setItem('auth_token', newToken);
  }, []);

  const signOut = useCallback(() => {
    setUser(null);
    setToken(null);
    localStorage.removeItem('auth_token');
  }, []);

  return (
    <AuthContext.Provider value={{ user, token, isLoading, signUp, signIn, signOut }}>
      {children}
    </AuthContext.Provider>
  );
}
