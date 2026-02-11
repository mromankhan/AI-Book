import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Part 0: Orientation & Setup',
      items: [
        'part-0-orientation/chapter-0.1-what-is-physical-ai',
        'part-0-orientation/chapter-0.2-learning-path-tooling'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Part 1: Robotics Fundamentals',
      items: [
        'part-1-robotics-fundamentals/chapter-1.1-anatomy-of-a-robot',
        'part-1-robotics-fundamentals/chapter-1.2-sensors-perception',
        'part-1-robotics-fundamentals/chapter-1.3-actuators-motion'
        // Additional chapters will be added as they are created
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Part 2: Physical AI Core Concepts',
      items: [
        'part-2-physical-ai-core/chapter-2.1-perception-in-physical-ai',
        'part-2-physical-ai-core/chapter-2.2-control-systems',
        'part-2-physical-ai-core/chapter-2.3-decision-making'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Part 3: Learning & Intelligence',
      items: [
        'part-3-learning-intelligence/chapter-3.1-machine-learning-for-robotics',
        'part-3-learning-intelligence/chapter-3.2-reinforcement-learning-basics'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Part 4: Humanoid Robotics',
      items: [
        'part-4-humanoid-robotics/chapter-4.1-what-makes-a-robot-humanoid',
        'part-4-humanoid-robotics/chapter-4.2-locomotion-balance'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Part 5: Simulation & Practice',
      items: [
        'part-5-simulation-practice/chapter-5.1-simulation-first-approach',
        'part-5-simulation-practice/chapter-5.2-hands-on-mini-projects'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Part 6: System Integration',
      items: [
        'part-6-system-integration/chapter-6.1-software-hardware-thinking',
        'part-6-system-integration/chapter-6.2-safety-ethics-future'
      ],
      collapsed: false,
    },
    {
      type: 'doc',
      id: 'glossary',
      label: 'Glossary of Terms'
    },
    {
      type: 'doc',
      id: 'index',
      label: 'Index'
    }
  ],
};

export default sidebars;
