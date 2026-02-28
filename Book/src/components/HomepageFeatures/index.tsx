import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  icon: string;
  description: ReactNode;
  gradient: string;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'AI-Powered Learning',
    icon: '\uD83E\uDD16',
    gradient: 'linear-gradient(135deg, #6366F1, #8B5CF6)',
    description: (
      <>
        Ask our RAG chatbot anything about the book. Get instant explanations,
        select text to dive deeper, and personalize content to your skill level.
      </>
    ),
  },
  {
    title: 'Comprehensive Curriculum',
    icon: '\uD83D\uDCDA',
    gradient: 'linear-gradient(135deg, #3B82F6, #06B6D4)',
    description: (
      <>
        16 chapters across 7 parts &mdash; from robot anatomy and sensors through
        reinforcement learning to humanoid locomotion and sim-to-real transfer.
      </>
    ),
  },
  {
    title: 'Interactive Experience',
    icon: '\u26A1',
    gradient: 'linear-gradient(135deg, #F59E0B, #EF4444)',
    description: (
      <>
        Translate chapters to Urdu, personalize difficulty, search across content,
        and explore hands-on simulation projects and mini-labs.
      </>
    ),
  },
];

function Feature({title, icon, description, gradient}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className={styles.featureCard}>
        <div className={styles.featureIconWrap} style={{background: gradient}}>
          <span className={styles.featureIcon}>{icon}</span>
        </div>
        <div className={styles.featureContent}>
          <Heading as="h3" className={styles.featureTitle}>{title}</Heading>
          <p className={styles.featureDesc}>{description}</p>
        </div>
        <div className={styles.featureGradientBorder} style={{background: gradient}} />
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2" className={styles.sectionTitle}>Why This Book?</Heading>
          <p className={styles.sectionSubtitle}>
            Built for the next generation of robotics engineers
          </p>
        </div>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
