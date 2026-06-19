// @ts-check
// Two sidebars: a task-oriented Usage guide and an Engineering wiki.

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  usage: [
    'intro',
    {
      type: 'category',
      label: 'Usage',
      collapsed: false,
      items: [
        'usage/installation',
        'usage/configuration',
        'usage/brokers',
        'usage/scanning',
        'usage/backtesting',
        'usage/live-trading',
        'usage/optimization',
        'usage/portfolio',
      ],
    },
  ],
  engineering: [
    {
      type: 'category',
      label: 'Engineering Wiki',
      collapsed: false,
      items: [
        'engineering/architecture',
        'engineering/separation-of-concerns',
        'engineering/broker-abstraction',
        'engineering/data-flow',
        'engineering/strategies',
        'engineering/scanners',
        'engineering/indicators',
        'engineering/engine',
        'engineering/optimization',
        'engineering/portfolio',
        'engineering/testing',
        'engineering/coding-standards',
        'engineering/extending',
      ],
    },
  ],
};

module.exports = sidebars;
