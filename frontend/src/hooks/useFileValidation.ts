export const ACCEPTED_MIME_TYPES = [
  'application/pdf',
  'image/png',
  'image/jpeg',
  'image/jpg',
];

export const ACCEPTED_EXTENSIONS = ['.pdf', '.png', '.jpg', '.jpeg'];

export const MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024; // 50 МБ — согласовано с бэкендом
export const MAX_FILES = 3;

export interface ValidationIssue {
  file: File;
  reason: string;
}

export interface ValidationResult {
  accepted: File[];
  rejected: ValidationIssue[];
}

function hasAcceptedExtension(name: string): boolean {
  const lower = name.toLowerCase();
  return ACCEPTED_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

export function validateFiles(incoming: File[], existing: File[] = []): ValidationResult {
  const accepted: File[] = [];
  const rejected: ValidationIssue[] = [];

  const remainingSlots = Math.max(0, MAX_FILES - existing.length);

  for (const file of incoming) {
    if (accepted.length >= remainingSlots) {
      rejected.push({
        file,
        reason: `Можно загрузить не более ${MAX_FILES} файлов за раз.`,
      });
      continue;
    }

    const validType =
      ACCEPTED_MIME_TYPES.includes(file.type) || hasAcceptedExtension(file.name);

    if (!validType) {
      rejected.push({
        file,
        reason: 'Неподдерживаемый формат. Допустимы только PDF, PNG и JPG.',
      });
      continue;
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      rejected.push({
        file,
        reason: `Файл больше ${(MAX_FILE_SIZE_BYTES / (1024 * 1024)).toFixed(0)} МБ.`,
      });
      continue;
    }

    if (file.size === 0) {
      rejected.push({ file, reason: 'Файл пустой.' });
      continue;
    }

    const isDuplicate =
      existing.some((f) => f.name === file.name && f.size === file.size) ||
      accepted.some((f) => f.name === file.name && f.size === file.size);

    if (isDuplicate) {
      rejected.push({ file, reason: 'Этот файл уже добавлен.' });
      continue;
    }

    accepted.push(file);
  }

  return { accepted, rejected };
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} Б`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} КБ`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} МБ`;
}
