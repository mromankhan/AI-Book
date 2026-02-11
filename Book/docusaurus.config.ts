import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'Physical AI: Humanoid & Robotics Systems',
  tagline: 'Comprehensive education in Physical AI and Robotics',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://mromankhan.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/AI-Book/',

  // GitHub pages deployment config.
  organizationName: 'mromankhan', // Usually your GitHub org/user name.
  projectName: 'AI-Book', // Usually your repo name.
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'throw',

  // Custom fields accessible via useDocusaurusContext
  customFields: {
    chatbotApiUrl: process.env.CHATBOT_API_URL || 'http://localhost:8000',
  },

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/mromankhan/AI-Book',
          showLastUpdateAuthor: true,
          showLastUpdateTime: true,
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/mromankhan/AI-Book',
          // Useful options to enforce blogging best practices
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  // Removing Algolia search for now to avoid validation errors
  // themes: [
  //   // For search functionality
  //   [
  //     '@docusaurus/theme-search-algolia',
  //     {
  //       // Configuration will go here but for now we'll keep it minimal to prevent validation errors
  //     },
  //   ],
  // ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Physical AI Book',
      logo: {
        alt: 'Physical AI Book Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Book Contents',
        },
        {to: '/blog', label: 'Blog', position: 'left'},
        {
          href: 'https://github.com/mromankhan/AI-Book',
          label: 'GitHub',
          position: 'right',
        },
        {
          type: 'custom-userMenu',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Book Sections',
          items: [
            {
              label: 'Part 0: Orientation',
              to: '/docs/part-0-orientation/chapter-0.1-what-is-physical-ai',
            },
            {
              label: 'Part 1: Robotics Fundamentals',
              to: '/docs/part-1-robotics-fundamentals/chapter-1.1-anatomy-of-a-robot',
            },
            {
              label: 'Part 2: Physical AI Core',
              to: '/docs/part-2-physical-ai-core/chapter-2.1-perception-in-physical-ai',
            },
            {
              label: 'Part 5: Simulation Practice',
              to: '/docs/part-5-simulation-practice/chapter-5.1-simulation-first-approach',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/mromankhan/AI-Book',
            },
            {
              label: 'Discord',
              href: 'https://discordapp.com/invite/physical-ai',
            },
            {
              label: 'X',
              href: 'https://x.com/physicalAI',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'Simulation Tools',
              to: '/docs/part-5-simulation-practice/chapter-5.1-simulation-first-approach',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/mromankhan/AI-Book',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI Education Project. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
