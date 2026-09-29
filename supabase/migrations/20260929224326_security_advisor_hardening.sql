create schema if not exists private;

revoke all on schema private from public;
grant usage on schema private to anon, service_role;

create or replace function private.match_document_chunks_impl(
    p_query_embedding extensions.vector(384),
    p_match_count integer default 15,
    p_embedding_model text default 'BAAI/bge-small-en-v1.5',
    p_chunking_version text default 'section-aware-v2'
)
returns table (
    chunk_id text,
    source_id text,
    source text,
    title text,
    page integer,
    section text,
    content text,
    metadata jsonb,
    similarity double precision,
    retrieval_methods text
)
language sql
stable
security definer
set search_path = ''
as $$
    select
        dc.chunk_id,
        dc.source_id,
        dc.source,
        dc.title,
        dc.page,
        dc.section,
        dc.content,
        dc.metadata,
        (1 - (dc.embedding OPERATOR(extensions.<=>) p_query_embedding))::double precision as similarity,
        'dense'::text as retrieval_methods
    from public.document_chunks dc
    where dc.embedding_model = p_embedding_model
      and dc.chunking_version = p_chunking_version
    order by dc.embedding OPERATOR(extensions.<=>) p_query_embedding
    limit greatest(p_match_count, 1);
$$;

create or replace function private.hybrid_search_document_chunks_impl(
    p_query_text text,
    p_query_embedding extensions.vector(384),
    p_match_count integer default 20,
    p_rrf_k integer default 60,
    p_embedding_model text default 'BAAI/bge-small-en-v1.5',
    p_chunking_version text default 'section-aware-v2'
)
returns table (
    chunk_id text,
    source_id text,
    source text,
    title text,
    page integer,
    section text,
    content text,
    metadata jsonb,
    similarity double precision,
    text_rank real,
    rrf_score double precision,
    retrieval_methods text
)
language sql
stable
security definer
set search_path = ''
as $$
with params as (
    select websearch_to_tsquery('english'::regconfig, p_query_text) as tsq
),
semantic as (
    select
        dc.chunk_id,
        (1 - (dc.embedding OPERATOR(extensions.<=>) p_query_embedding))::double precision as similarity,
        row_number() over (
            order by dc.embedding OPERATOR(extensions.<=>) p_query_embedding
        ) as semantic_rank
    from public.document_chunks dc
    where dc.embedding_model = p_embedding_model
      and dc.chunking_version = p_chunking_version
    order by dc.embedding OPERATOR(extensions.<=>) p_query_embedding
    limit greatest(p_match_count * 3, 30)
),
keyword as (
    select
        dc.chunk_id,
        ts_rank_cd(dc.fts, params.tsq)::real as text_rank,
        row_number() over (
            order by ts_rank_cd(dc.fts, params.tsq) desc
        ) as keyword_rank
    from public.document_chunks dc
    cross join params
    where dc.embedding_model = p_embedding_model
      and dc.chunking_version = p_chunking_version
      and dc.fts @@ params.tsq
    order by ts_rank_cd(dc.fts, params.tsq) desc
    limit greatest(p_match_count * 3, 30)
),
fused as (
    select
        coalesce(s.chunk_id, k.chunk_id) as chunk_id,
        s.similarity,
        k.text_rank,
        (
            case
                when s.semantic_rank is not null
                then 1.0 / (greatest(p_rrf_k, 1) + s.semantic_rank)
                else 0
            end
            +
            case
                when k.keyword_rank is not null
                then 1.0 / (greatest(p_rrf_k, 1) + k.keyword_rank)
                else 0
            end
        )::double precision as rrf_score,
        concat_ws(
            '+',
            case when s.semantic_rank is not null then 'dense' end,
            case when k.keyword_rank is not null then 'fts' end
        ) as retrieval_methods
    from semantic s
    full outer join keyword k on s.chunk_id = k.chunk_id
)
select
    dc.chunk_id,
    dc.source_id,
    dc.source,
    dc.title,
    dc.page,
    dc.section,
    dc.content,
    dc.metadata,
    f.similarity,
    f.text_rank,
    f.rrf_score,
    f.retrieval_methods
from fused f
join public.document_chunks dc on dc.chunk_id = f.chunk_id
order by f.rrf_score desc
limit greatest(p_match_count, 1);
$$;

create or replace function private.telecom_rag_status_impl(
    p_embedding_model text default 'BAAI/bge-small-en-v1.5',
    p_chunking_version text default 'section-aware-v2'
)
returns jsonb
language sql
stable
security definer
set search_path = ''
as $$
    select jsonb_build_object(
        'document_chunks_total', (select count(*) from public.document_chunks),
        'document_chunks_current', (
            select count(*)
            from public.document_chunks
            where embedding_model = p_embedding_model
              and chunking_version = p_chunking_version
        ),
        'kpi_observations', (select count(*) from public.kpi_observations),
        'embedding_model', p_embedding_model,
        'chunking_version', p_chunking_version
    );
$$;

revoke all on function private.match_document_chunks_impl(extensions.vector, integer, text, text)
from public, anon, authenticated;
revoke all on function private.hybrid_search_document_chunks_impl(text, extensions.vector, integer, integer, text, text)
from public, anon, authenticated;
revoke all on function private.telecom_rag_status_impl(text, text)
from public, anon, authenticated;

grant execute on function private.match_document_chunks_impl(extensions.vector, integer, text, text)
to anon, service_role;
grant execute on function private.hybrid_search_document_chunks_impl(text, extensions.vector, integer, integer, text, text)
to anon, service_role;
grant execute on function private.telecom_rag_status_impl(text, text)
to anon, service_role;

create or replace function public.match_document_chunks(
    p_query_embedding extensions.vector(384),
    p_match_count integer default 15,
    p_embedding_model text default 'BAAI/bge-small-en-v1.5',
    p_chunking_version text default 'section-aware-v2'
)
returns table (
    chunk_id text,
    source_id text,
    source text,
    title text,
    page integer,
    section text,
    content text,
    metadata jsonb,
    similarity double precision,
    retrieval_methods text
)
language sql
stable
security invoker
set search_path = ''
as $$
    select *
    from private.match_document_chunks_impl(
        p_query_embedding,
        p_match_count,
        p_embedding_model,
        p_chunking_version
    );
$$;

create or replace function public.hybrid_search_document_chunks(
    p_query_text text,
    p_query_embedding extensions.vector(384),
    p_match_count integer default 20,
    p_rrf_k integer default 60,
    p_embedding_model text default 'BAAI/bge-small-en-v1.5',
    p_chunking_version text default 'section-aware-v2'
)
returns table (
    chunk_id text,
    source_id text,
    source text,
    title text,
    page integer,
    section text,
    content text,
    metadata jsonb,
    similarity double precision,
    text_rank real,
    rrf_score double precision,
    retrieval_methods text
)
language sql
stable
security invoker
set search_path = ''
as $$
    select *
    from private.hybrid_search_document_chunks_impl(
        p_query_text,
        p_query_embedding,
        p_match_count,
        p_rrf_k,
        p_embedding_model,
        p_chunking_version
    );
$$;

create or replace function public.telecom_rag_status(
    p_embedding_model text default 'BAAI/bge-small-en-v1.5',
    p_chunking_version text default 'section-aware-v2'
)
returns jsonb
language sql
stable
security invoker
set search_path = ''
as $$
    select private.telecom_rag_status_impl(
        p_embedding_model,
        p_chunking_version
    );
$$;

revoke all on function public.match_document_chunks(extensions.vector, integer, text, text)
from public, authenticated;
revoke all on function public.hybrid_search_document_chunks(text, extensions.vector, integer, integer, text, text)
from public, authenticated;
revoke all on function public.telecom_rag_status(text, text)
from public, authenticated;

grant execute on function public.match_document_chunks(extensions.vector, integer, text, text)
to anon, service_role;
grant execute on function public.hybrid_search_document_chunks(text, extensions.vector, integer, integer, text, text)
to anon, service_role;
grant execute on function public.telecom_rag_status(text, text)
to anon, service_role;

alter function public.cloudflare_usage_today() security invoker;
alter function public.record_cloudflare_usage(text, integer, integer) security invoker;

revoke all on function public.cloudflare_usage_today()
from public, anon, authenticated;
revoke all on function public.record_cloudflare_usage(text, integer, integer)
from public, anon, authenticated;

grant execute on function public.cloudflare_usage_today() to service_role;
grant execute on function public.record_cloudflare_usage(text, integer, integer) to service_role;

grant select, insert on table public.llm_usage to service_role;
grant usage, select on sequence public.llm_usage_id_seq to service_role;

revoke all on table public.document_chunks from anon, authenticated;
revoke all on table public.llm_usage from anon, authenticated;

drop policy if exists "deny direct document chunk access" on public.document_chunks;
create policy "deny direct document chunk access"
on public.document_chunks
as restrictive
for all
to anon, authenticated
using (false)
with check (false);

drop policy if exists "deny direct llm usage access" on public.llm_usage;
create policy "deny direct llm usage access"
on public.llm_usage
as restrictive
for all
to anon, authenticated
using (false)
with check (false);

revoke execute on function public.rls_auto_enable()
from public, anon, authenticated, service_role;

revoke truncate, trigger, references on table public.kpi_observations
from anon, authenticated;
