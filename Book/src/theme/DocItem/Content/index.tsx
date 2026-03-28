import React, { JSX } from 'react';
import OriginalDocItemContent from '@theme-original/DocItem/Content';
import { useDoc } from '@docusaurus/plugin-content-docs/client';
import ChapterToolbar from '@site/src/components/ChapterActions/ChapterToolbar';
import type { WrapperProps } from '@docusaurus/types';
import type DocItemContentType from '@theme/DocItem/Content';

type Props = WrapperProps<typeof DocItemContentType>;

export default function DocItemContent(props: Props): JSX.Element {
  const { metadata } = useDoc();
  const isChapter = metadata.id?.includes('chapter');

  return (
    <>
      {isChapter && (
        <ChapterToolbar chapter={metadata.title} />
      )}
      <OriginalDocItemContent {...props} />
    </>
  );
}
