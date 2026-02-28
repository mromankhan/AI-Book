import type {ReactNode} from 'react';
import {useEffect, useState} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';

import styles from './index.module.css';

/* -------- Typing animation hook -------- */
function useTypingEffect(texts: string[], speed = 80, pause = 2000) {
  const [displayed, setDisplayed] = useState('');
  const [textIndex, setTextIndex] = useState(0);
  const [charIndex, setCharIndex] = useState(0);
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    const current = texts[textIndex];
    let timeout: ReturnType<typeof setTimeout>;

    if (!isDeleting && charIndex < current.length) {
      timeout = setTimeout(() => {
        setDisplayed(current.slice(0, charIndex + 1));
        setCharIndex(c => c + 1);
      }, speed);
    } else if (!isDeleting && charIndex === current.length) {
      timeout = setTimeout(() => setIsDeleting(true), pause);
    } else if (isDeleting && charIndex > 0) {
      timeout = setTimeout(() => {
        setDisplayed(current.slice(0, charIndex - 1));
        setCharIndex(c => c - 1);
      }, speed / 2);
    } else if (isDeleting && charIndex === 0) {
      setIsDeleting(false);
      setTextIndex((textIndex + 1) % texts.length);
    }

    return () => clearTimeout(timeout);
  }, [charIndex, isDeleting, textIndex, texts, speed, pause]);

  return displayed;
}

/* -------- Hero Section -------- */
function HeroSection() {
  const typedText = useTypingEffect([
    'Humanoid Robotics',
    'Perception & Control',
    'Reinforcement Learning',
    'Simulation & Practice',
  ]);

  return (
    <header className={styles.hero}>
      <div className={styles.heroDecor}>
        <div className={styles.decorCircle1} />
        <div className={styles.decorCircle2} />
        <div className={styles.decorCircle3} />
        <div className={styles.gridOverlay} />
      </div>

      <div className={clsx('container', styles.heroContent)}>
        <div className={styles.heroBadge}>
          <span className={styles.badgeDot} />
          Open Source & AI-Powered
        </div>

        <Heading as="h1" className={styles.heroTitle}>
          Physical AI &{' '}
          <span className={styles.gradientText}>Robotics Systems</span>
        </Heading>

        <p className={styles.heroSubtitle}>
          A comprehensive, interactive textbook on{' '}
          <span className={styles.typedText}>{typedText}</span>
          <span className={styles.cursor}>|</span>
        </p>

        <div className={styles.heroButtons}>
          <Link
            className={clsx('button button--lg', styles.primaryButton)}
            to="/docs/part-0-orientation/chapter-0.1-what-is-physical-ai">
            Start Reading
          </Link>
          <Link
            className={clsx('button button--lg', styles.secondaryButton)}
            to="/docs/intro">
            Explore Chapters
          </Link>
        </div>
      </div>
    </header>
  );
}

/* -------- Stats Section -------- */
const stats = [
  { value: '7', label: 'Parts', icon: '\uD83D\uDCDA' },
  { value: '16', label: 'Chapters', icon: '\uD83D\uDCD6' },
  { value: 'AI', label: 'Powered Learning', icon: '\uD83E\uDD16' },
  { value: '\u221E', label: 'Curiosity', icon: '\uD83D\uDE80' },
];

function StatsSection() {
  return (
    <section className={styles.statsSection}>
      <div className="container">
        <div className={styles.statsGrid}>
          {stats.map((stat, i) => (
            <div key={i} className={styles.statCard}>
              <span className={styles.statIcon}>{stat.icon}</span>
              <span className={styles.statValue}>{stat.value}</span>
              <span className={styles.statLabel}>{stat.label}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* -------- Topics Grid -------- */
const topics = [
  { icon: '\uD83E\uDDBE', title: 'Robot Anatomy', desc: 'Joints, links, kinematics & DOF' },
  { icon: '\uD83D\uDC41\uFE0F', title: 'Perception', desc: 'Vision, LIDAR, sensor fusion' },
  { icon: '\uD83C\uDFAE', title: 'Control Systems', desc: 'PID, feedback loops, trajectory' },
  { icon: '\uD83E\uDDE0', title: 'Machine Learning', desc: 'Supervised, RL, neural networks' },
  { icon: '\uD83E\uDDB6', title: 'Locomotion', desc: 'Bipedal walking, balance control' },
  { icon: '\uD83C\uDF10', title: 'Simulation', desc: 'Gazebo, digital twins, sim-to-real' },
  { icon: '\u2699\uFE0F', title: 'Integration', desc: 'Software-hardware, ROS 2' },
  { icon: '\uD83D\uDEE1\uFE0F', title: 'Safety & Ethics', desc: 'Responsible AI, human-robot interaction' },
];

function TopicsSection() {
  return (
    <section className={styles.topicsSection}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2" className={styles.sectionTitle}>What You'll Learn</Heading>
          <p className={styles.sectionSubtitle}>
            From foundational concepts to advanced robotics &mdash; everything in one place
          </p>
        </div>
        <div className={styles.topicsGrid}>
          {topics.map((topic, i) => (
            <div key={i} className={styles.topicCard}>
              <span className={styles.topicIcon}>{topic.icon}</span>
              <h3 className={styles.topicTitle}>{topic.title}</h3>
              <p className={styles.topicDesc}>{topic.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* -------- CTA Section -------- */
function CTASection() {
  return (
    <section className={styles.ctaSection}>
      <div className="container">
        <div className={styles.ctaCard}>
          <Heading as="h2" className={styles.ctaTitle}>
            Ready to Build the Future?
          </Heading>
          <p className={styles.ctaText}>
            Dive into Physical AI with our AI-powered learning assistant by your side.
            Ask questions, get explanations, and personalize your learning journey.
          </p>
          <Link
            className={clsx('button button--lg', styles.ctaButton)}
            to="/docs/part-0-orientation/chapter-0.1-what-is-physical-ai">
            Begin Chapter One
          </Link>
        </div>
      </div>
    </section>
  );
}

/* -------- Main Page -------- */
export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title="Home"
      description="Comprehensive education in Physical AI and Robotics - an interactive, AI-powered textbook">
      <HeroSection />
      <StatsSection />
      <main>
        <HomepageFeatures />
        <TopicsSection />
      </main>
      <CTASection />
    </Layout>
  );
}
